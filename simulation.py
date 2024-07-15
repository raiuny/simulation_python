import simpy
import random
from time import time
import numpy as np
from typing import List
from arrival_model import IntervalGeneratorFactory, ArrivalType
from packet import Pkt
import math
from multiprocessing import Pool
import argparse
import pandas as pd
import threading

class SafeCounter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def increment(self):
        with self.lock:
            self.value += 1

    def decrement(self):
        with self.lock:
            self.value -= 1

    def get_value(self):
        with self.lock:
            return self.value


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam1", type=float, help="arrival rate of mld", default=0.004)
    ap.add_argument("--lam21", type=float, help="arrival rate of sld", default=0.001)
    ap.add_argument("--lam22", type=float, help="arrival rate of sld", default=0.002)
    ap.add_argument("--beta", type=float, help="beta", default=0.5)
    ap.add_argument("--nmld", type=int, default=10)
    ap.add_argument("--nsld1", type=int, default=10)
    ap.add_argument("--nsld2", type=int, default=10)
    ap.add_argument("--sldW", type=int, default=16)
    ap.add_argument("--sldK", type=int, default=6)
    ap.add_argument("--mldW1", type=int, default=16)
    ap.add_argument("--mldW2", type=int, default=16)
    ap.add_argument("--mldK1", type=int, default=6)
    ap.add_argument("--mldK2", type=int, default=6)
    ap.add_argument("--rt", type=int, default=1e6)
    ap.add_argument("--st", type=int, default=-1)
    ap.add_argument("--snum", type=int, default=1)
    ap.add_argument("--path", type=str, default="default.csv")
    return ap.parse_args()

class Params(object):
    sim_num = 1
    save_path = "default.csv"
    
    arrival_rate_mld = 0.004 # per node per slot
    arrival_rate_sld1 = 0.002 # per node per slot
    arrival_rate_sld2 = 0.002 # per node per slot
    nlink = 2 # 2 links
    beta = 0.5
    nmld = 10
    nsld1 = 10
    nsld2 = 10
    mldW = 16
    mldK = 6 
    sldW = 16
    sldK = 6
    sim_duration = 1e7
    start_time = sim_duration / 4
    tt = 32
    tf = 27
    
    # MLD
    queueing_time_link = [[] for _ in range(nlink)]
    access_time_link = [[] for _ in range(nlink)]
    access_time_sec_link = [[] for _ in range(nlink)]
    e2e_time_link = [[] for _ in range(nlink)]
    thpt_link = [0 for _ in range(nlink)]
    suc_link = [0 for _ in range(nlink)]
    col_link = [0 for _ in range(nlink)]
    
    
    # SLD
    queueing_time_link_sld = [[] for _ in range(nlink)]
    access_time_link_sld = [[] for _ in range(nlink)]
    access_time_sec_link_sld = [[] for _ in range(nlink)]
    e2e_time_link_sld = [[] for _ in range(nlink)]
    thpt_link_sld = [0 for _ in range(nlink)]
    suc_link_sld = [0 for _ in range(nlink)]
    col_link_sld = [0 for _ in range(nlink)]
    
    # total
    alpha_link = [0 for _ in range(nlink)]
    fin_counter = 0
    pkts_counter = 0
    
    sfcs = [SafeCounter() for _ in range(nlink)] # link上的申请人数


class MLD(object):
    def __init__(self, id: int, env: simpy.Environment, arr_type: ArrivalType, lam, init_w, cut_stg, suc_time, col_time, beta):
        self.id = id
        self.type = (beta < 0) or (beta >= 1)
        self.env = env      
        # self.action = start_delayed(env, self.run(), )
        self.beta = beta
        self.init_w = init_w
        self.cut_stg = cut_stg
        self.suc_time = suc_time
        self.col_time = col_time
        self.max_w = init_w * math.pow(2, cut_stg)
        self.boc_rngs = []
        self.bows = np.zeros(Params.nlink)
        self.bocs = np.zeros(Params.nlink)
        for i in range(Params.nlink):
            self.bows[i] = init_w
            self.boc_rngs.append(np.random.RandomState())
            self.reset_boc(i)

        self.arr_itv_generator = IntervalGeneratorFactory.create(arr_type, lam=lam)
        self.pkt_num = 0
        self.pkts_on_link = [[] for _ in range(Params.nlink)]
    
    def arrival_interval(self):
        return self.arr_itv_generator.get_itv()
    
    def run(self):
        self.env.process(self.generate_pkts())
        for i in range(Params.nlink):
            self.env.process(self.try_connecting(i))
        
    def generate_pkts(self):
        while True:
            itv = self.arrival_interval()
            yield self.env.timeout(itv)
            self.pkt_num += 1
            pkt = Pkt(self.id, self.env.now,num=self.pkt_num)
            if self.env.now > Params.start_time:
                Params.pkts_counter += 1
            self.allocating(pkt)
            
    def allocating(self, pkt):
        rv = random.uniform(0, 1)
        if rv < self.beta:
            self.pkts_on_link[0].append(pkt)
            if self.pkts_on_link[0][0].ser_time == -1:
                self.pkts_on_link[0][0].ser_time = self.env.now
        else:
            self.pkts_on_link[1].append(pkt)
            if self.pkts_on_link[1][0].ser_time == -1:
                self.pkts_on_link[1][0].ser_time = self.env.now
                    
    def try_connecting(self, linkid):
        while True:
            if len(self.pkts_on_link[linkid]) > 0:
                assert self.pkts_on_link[linkid][0].ser_time!=-1, "HOL包无开始服务的时间"
                if self.bocs[linkid] == 0:
                    yield self.env.timeout(0.5)
                    if Params.sfcs[linkid].get_value() == 0:
                        yield self.env.timeout(0.5)
                        Params.sfcs[linkid].increment()
                        yield self.env.timeout(0.5)
                        assert Params.sfcs[linkid].get_value() > 0, ">0"
                        if Params.sfcs[linkid].get_value() == 1:
                            yield self.env.timeout(self.suc_time-0.5)
                            self.reset_bow(linkid, 0)
                            self.reset_boc(linkid)
                            pkt = self.pkts_on_link[linkid].pop(0)
                            pkt.dep_time = self.env.now
                            if self.env.now > Params.start_time:
                                if self.type: # SLD
                                    Params.queueing_time_link_sld[linkid].append(pkt.ser_time - pkt.arr_time) 
                                    Params.access_time_link_sld[linkid].append(pkt.dep_time - pkt.ser_time)
                                    Params.access_time_sec_link_sld[linkid].append((pkt.dep_time - pkt.ser_time)**2)
                                    Params.e2e_time_link_sld[linkid].append(pkt.dep_time - pkt.arr_time)
                                    Params.fin_counter += 1
                                    Params.thpt_link_sld[linkid] += 1
                                    Params.suc_link_sld[linkid] += 1
                                else: # MLD
                                    Params.queueing_time_link[linkid].append(pkt.ser_time - pkt.arr_time) 
                                    Params.access_time_link[linkid].append(pkt.dep_time - pkt.ser_time)
                                    Params.access_time_sec_link[linkid].append((pkt.dep_time - pkt.ser_time)**2)
                                    Params.e2e_time_link[linkid].append(pkt.dep_time - pkt.arr_time)
                                    Params.fin_counter += 1
                                    Params.thpt_link[linkid] += 1
                                    Params.suc_link[linkid] += 1
                            if len(self.pkts_on_link[linkid]) > 0 and self.pkts_on_link[linkid][0].ser_time == -1:
                                self.pkts_on_link[linkid][0].ser_time = self.env.now
                        else: # 冲突
                            yield self.env.timeout(self.col_time-0.5)
                            self.reset_bow(linkid, 1)
                            self.reset_boc(linkid)
                            if self.env.now > Params.start_time:
                                if self.type: # SLD
                                    Params.col_link_sld[linkid] += 1
                                else:
                                    Params.col_link[linkid] += 1
                        Params.sfcs[linkid].decrement()
                    else:
                        yield self.env.timeout(0.5)
                else:
                    yield self.env.timeout(0.2)
                    if Params.sfcs[linkid].get_value() == 0:
                        self.bocs[linkid] -= 1
                        yield self.env.timeout(0.8)
                    else:
                        yield self.env.timeout(0.8)
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
        for i in range(Params.nmld):
            self.mlds.append(MLD(i, self.env, ArrivalType.BERNOULLI, Params.arrival_rate_mld, Params.mldW, Params.mldK, Params.tt, Params.tf, beta = Params.beta))
        # SLD
        for i in range(Params.nsld1):
            self.mlds.append(MLD(i+Params.nmld, self.env, ArrivalType.BERNOULLI, Params.arrival_rate_sld1, Params.sldW, Params.sldK, Params.tt, Params.tf, beta = 1))
        for i in range(Params.nsld2):
            self.mlds.append(MLD(i+Params.nmld+Params.nsld1, self.env, ArrivalType.BERNOULLI, Params.arrival_rate_sld2, Params.sldW, Params.sldK, Params.tt, Params.tf, beta = -0.1))
        # 优先资源: PriorityResource 一般资源: Resource (FCFS)
            
    def run(self):
        for i in range(Params.nlink):
            self.env.process(self.step_process(i))
        for mld in self.mlds:
            mld.run()
        self.env.run(until=Params.sim_duration)
    
    def step_process(self, i):
        while True:
            if self.env.now > Params.start_time:
                if Params.sfcs[i].get_value() == 0:
                    yield self.env.timeout(0.5)
                    if Params.sfcs[i].get_value() == 0:
                        Params.alpha_link[i] += 1
                    yield self.env.timeout(0.5)
                else:
                    yield self.env.timeout(1)
            else:
                yield self.env.timeout(1)
            

    def get_result(self):
        df = {}
        # print("Throughput:\n(MLD)")
        for i in range(Params.nlink):
            df[f"Throughput of MLD on Link {i+1}"] = Params.thpt_link[i] / (Params.sim_duration - Params.start_time)
        # print("\n(SLD)")
        for i in range(Params.nlink):
            df[f"Throughput of SLD on Link {i+1}"] = Params.thpt_link_sld[i] / (Params.sim_duration - Params.start_time)
        # print()
        # print("\nDelay:\n(MLD)")
        # print("Access Delay:")
        for i in range(Params.nlink):
            if len(Params.access_time_link[i]) == 0:
                df[f"Access Delay of MLD on Link {i+1}"] = np.nan
            else:
                df[f"Access Delay of MLD on Link {i+1}"] = np.mean(Params.access_time_link[i])
        # print("\nQueueing Delay:")
        for i in range(Params.nlink):
            if len(Params.queueing_time_link[i]) == 0:
                df[f"Queueing Delay of MLD on Link {i+1}"] = np.nan
            else:
                df[f"Queueing Delay of MLD on Link {i+1}"] = np.mean(Params.queueing_time_link[i])
        # print()
        # print("\n(SLD)")
        # print("Access Delay:")
        for i in range(Params.nlink):
            if len(Params.access_time_link_sld[i]) == 0:
                df[f"Access Delay of SLD on Link {i+1}"] = np.nan
            else:
                df[f"Access Delay of SLD on Link {i+1}"] = np.mean(Params.access_time_link_sld[i])
        # print("\nQueueing Delay:")
        for i in range(Params.nlink):
            if len(Params.queueing_time_link_sld[i]) == 0:
                df[f"Queueing Delay of SLD on Link {i+1}"] = np.nan
            else:
                df[f"Queueing Delay of SLD on Link {i+1}"] = np.mean(Params.queueing_time_link_sld[i])
        # print()
        # print("\nsuccess transmit prob:")
        # print("(MLD)")
        for i in range(Params.nlink):
            if (Params.suc_link[i] + Params.col_link[i]) == 0:
                df[f"p of MLD on Link {i+1}"] = np.nan
            else:
                df[f"p of MLD on Link {i+1}"] = Params.suc_link[i] / (Params.suc_link[i] + Params.col_link[i])
        # print("\n(SLD)")
        for i in range(Params.nlink):
            if (Params.suc_link_sld[i] + Params.col_link_sld[i]) == 0:
                df[f"p of SLD on Link {i+1}"] = np.nan
            else:
                df[f"p of SLD on Link {i+1}"] = Params.suc_link_sld[i] / (Params.suc_link_sld[i] + Params.col_link_sld[i])
        # print()
        df[f"success transmit ratio"] = Params.fin_counter / Params.pkts_counter
        df["weighted e2e delay of mld"] = np.mean(Params.e2e_time_link[0]) * Params.beta + np.mean(Params.e2e_time_link[1]) * (1-Params.beta)
        df["alpha of link 1"] = Params.alpha_link[0] / (Params.sim_duration - Params.start_time)
        df["alpha of link 2"] = Params.alpha_link[1] / (Params.sim_duration - Params.start_time)
        for i in range(Params.nlink):
             df[f"sec access delay of mld on Link {i}"] = np.mean(Params.access_time_sec_link[0])
        for i in range(Params.nlink):
             df[f"sec access delay of sld on Link {i}"] = np.mean(Params.access_time_sec_link_sld[0])
        return df

def clear_log():
    nlink = Params.nlink
    Params.sfcs = [SafeCounter() for _ in range(nlink)]
    
    Params.queueing_time_link = [[] for _ in range(nlink)]
    Params.access_time_link = [[] for _ in range(nlink)]
    Params.access_time_sec_link = [[] for _ in range(nlink)]
    Params.e2e_time_link = [[] for _ in range(nlink)]
    Params.thpt_link = [0 for _ in range(nlink)]
    Params.suc_link = [0 for _ in range(nlink)]
    Params.col_link = [0 for _ in range(nlink)]
    
    
    # SLD
    Params.queueing_time_link_sld = [[] for _ in range(nlink)]
    Params.access_time_link_sld = [[] for _ in range(nlink)]
    Params.access_time_sec_link_sld = [[] for _ in range(nlink)]
    Params.e2e_time_link_sld = [[] for _ in range(nlink)]
    Params.thpt_link_sld = [0 for _ in range(nlink)]
    Params.suc_link_sld = [0 for _ in range(nlink)]
    Params.col_link_sld = [0 for _ in range(nlink)]
    # total
    Params.alpha_link = [0 for _ in range(nlink)]
    Params.fin_counter = 0
    Params.pkts_counter = 0
    
if __name__ == "__main__":
    begin = time()
    args = parse_args()
    print(args)
    Params.arrival_rate_mld = args.lam1
    Params.arrival_rate_sld1 = args.lam21
    Params.arrival_rate_sld2 = args.lam22
    Params.nmld = args.nmld
    Params.nsld1 = args.nsld1
    Params.nsld2 = args.nsld2
    Params.beta = args.beta
    Params.mldW = args.mldW1
    Params.sldW = args.sldW
    Params.mldK = args.mldK1
    Params.sim_duration = args.rt
    Params.start_time = args.st
    Params.sim_num = args.snum
    Params.save_path = args.path
    
    if args.st == -1:
        Params.start_time = args.rt/4
    np.random.seed(2024)
    res = pd.DataFrame()
    for i in range(Params.sim_num):
        sys = System()
        sys.run()
        end = time()
        df = sys.get_result()
        df = pd.DataFrame(df, index=[0])
        clear_log()
        if i == 0:
            res = df
        else:
            res = pd.concat([res, df], ignore_index=True)
        
    res.to_csv(Params.save_path)
    print(f"Total time: {end-begin:.4f} sec.")