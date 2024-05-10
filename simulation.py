import simpy
import random
from time import time
import numpy as np
from typing import List

from packet import Pkt
from arrival_model import ArrivalType

from simpy.util import start_delayed
from arrival_model import IntervalGeneratorFactory, ArrivalType
from packet import Pkt
import math

import argparse

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam1", type=float, help="lambda1", default=0.002)
    ap.add_argument("--lam2", type=float, help="lambda2", default=0.002)
    ap.add_argument("--nmld", type=int, default=10)
    ap.add_argument("--nsld1", type=int, default=0)
    ap.add_argument("--nsld2", type=int, default=0)
    return ap.parse_args()

class Params(object):
    arrival_rate = 0.002 # per node per slot
    nlink = 2 # 2 links
    nmld = 10
    nsld = 0
    W = 256
    K = 6 
    sim_duration = 1e6
    queuing_time_link = [[] for _ in range(nlink)]
    access_time_link = [[] for _ in range(nlink)]
    e2e_time_link = [[] for _ in range(nlink)]
    thpt_link = [0 for _ in range(nlink)]
    fin_counter = 0
    pkts_counter = 0
    tt = 32
    tf = 27
    beta = 1

class MLD(object):
    def __init__(self, id: int, env: simpy.Environment, links: List[simpy.Resource], arr_type: ArrivalType, lam, init_w, cut_stg, suc_time, col_time, beta):
        self.id = id
        self.env = env      
        # self.action = start_delayed(env, self.run(), )
        self.beta = beta
        self.init_w = init_w
        self.cut_stg = cut_stg
        self.suc_time = suc_time
        self.col_time = col_time
        self.max_w = init_w * math.pow(2, cut_stg)
        self.boc_rngs = []
        self.bows = np.zeros(len(links))
        self.bocs = np.zeros(len(links))
        self.links: List[simpy.Resource] = links
        for i in range(len(links)):
            self.bows[i] = init_w
            self.boc_rngs.append(np.random.RandomState())
            self.bocs[i] = 0

        self.arr_itv_generator = IntervalGeneratorFactory.create(arr_type, lam=lam)
        self.pkt_num = 0
        self.pkts_on_link = [[] for _ in range(len(links))]
    
    def arrival_interval(self):
        return self.arr_itv_generator.get_itv()
    
    def run(self):
        self.env.process(self.generate_pkts())
        for i in range(len(self.links)):
            self.env.process(self.try_connecting(i))
        
    def generate_pkts(self):
        while True:
            itv = self.arrival_interval()
            yield self.env.timeout(itv)
            self.pkt_num += 1
            pkt = Pkt(self.id, self.env.now,)
            Params.pkts_counter += 1
            self.allocating(pkt) 
            
    def allocating(self, pkt):
            rv = random.uniform(0, 1)
            if rv < self.beta:
                self.pkts_on_link[0].append(pkt)
            else:
                self.pkts_on_link[1].append(pkt)
            if len(self.pkts_on_link[0]) > 0 and self.pkts_on_link[0][0].ser_time == -1:
                self.pkts_on_link[0][0].ser_time = self.env.now
            if len(self.pkts_on_link[1]) > 0 and self.pkts_on_link[1][0].ser_time == -1:
                self.pkts_on_link[1][0].ser_time = self.env.now
                        
    def try_connecting(self, linkid):
        while True:
            if self.links[linkid].count == 0:
                if len(self.pkts_on_link[linkid]) > 0:
                    assert self.pkts_on_link[linkid][0].ser_time!=-1, "HOL包无开始服务的时间"
                    if self.bocs[linkid] == 0:
                        with self.links[linkid].request() as req:
                            if not req.triggered:
                                yield self.env.timeout(self.col_time)
                                self.reset_bow(linkid, 1)
                                self.reset_boc(linkid)
                            else:
                                yield self.env.timeout(self.suc_time)
                                self.reset_bow(linkid, 0)
                                self.reset_boc(linkid)
                                # print(self.pkts_on_link)
                                pkt = self.pkts_on_link[linkid].pop(0)
                                pkt.dep_time = self.env.now
                                Params.queuing_time_link[linkid].append(pkt.ser_time - pkt.arr_time) 
                                Params.access_time_link[linkid].append(pkt.dep_time - pkt.ser_time)
                                Params.e2e_time_link[linkid].append(pkt.dep_time - pkt.arr_time)
                                Params.fin_counter += 1
                                Params.thpt_link[linkid] += 1
                                if len(self.pkts_on_link[linkid]) > 0 and self.pkts_on_link[linkid][0].ser_time == -1:
                                    self.pkts_on_link[linkid][0].ser_time = self.env.now
                    else:
                        yield self.env.timeout(1)
                        self.bocs[linkid] -= 1
                else:
                    yield self.env.timeout(1)
                    self.bocs[linkid] = self.bocs[linkid] - 1 if self.bocs[linkid] > 0 else 0
            else:
                yield self.env.timeout(1)
            
                    
    
    def reset_bow(self, link_idx, flag = 0):
        if flag == 0:
            self.bows[link_idx] = self.init_w
        else:
            self.bows[link_idx] = min(self.bows[link_idx] * 2, self.max_w)
            
    def reset_boc(self, link_idx):
        self.bocs[link_idx] = self.boc_rngs[link_idx].randint(0, self.bows[link_idx])
    
        
class System(object):
    def __init__(self):
        self.env = simpy.Environment()
        
        # MLD
        self.mlds: List[MLD] = []
        self.links: List[simpy.Resource] = []
        for i in range(Params.nlink):
            self.links.append(simpy.Resource(self.env, capacity=1)) # 一般资源 先进先出
        for i in range(Params.nmld):
            self.mlds.append(MLD(i, self.env, self.links, ArrivalType.BERNOULLI, Params.arrival_rate, Params.W, Params.K, Params.tt, Params.tf, beta = Params.beta))
        # 优先资源: PriorityResource 一般资源: Resource (FCFS)
            
    def run(self):
        # self.env.process(self.step_process())
        for mld in self.mlds:
            mld.run()
        self.env.run(until=Params.sim_duration)
        return np.mean(Params.queuing_time_link[0]), np.mean(Params.access_time_link[0]), np.mean(Params.e2e_time_link[0])
    
    def step_process(self):
        yield self.env.timeout(5)
        print(self.env.now, np.mean(Params.queuing_time_link[0]), np.mean(Params.access_time_link[0]), np.mean(Params.e2e_time_link[0]), np.mean(Params.e2e_time_link[1]))

if __name__ == "__main__":
    args = parse_args()
    print(args)
    sys = System()
    qd, ad, ed = sys.run()
    run_time = Params.sim_duration
    print("result", qd, ad, ed)
    served_num, total_num = Params.fin_counter, Params.pkts_counter
    total = 0
    for mld in sys.mlds:
        total += mld.pkt_num
    print(served_num, total_num, total)
    print("throughput on link1: ", Params.thpt_link[0]/Params.sim_duration)
    print(Params.arrival_rate * Params.sim_duration)
