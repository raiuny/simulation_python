from simpy import Environment, Resource
from simpy.util import start_delayed
from enum import Enum
from arrival_model import IntervalGeneratorFactory, ArrivalType
import numpy as np
from packet import Pkt
from typing import List
import random


class MLD(object):
    def __init__(self, id: int, env: Environment, links: List[Resource], arr_type: ArrivalType, init_w, cut_stg, suc_time, col_time):
        self.id = id
        self.env = env      
        # self.action = start_delayed(env, self.run(), )
        self.beta = 1
        self.init_w = init_w
        self.cut_stg = cut_stg
        self.suc_time = suc_time
        self.col_time = col_time
        self.max_w = init_w * np.pow(2, cut_stg)
        self.boc_rngs = []
        self.bows = np.zeros(len(links))
        self.bocs = np.zeros(len(links))
        self.links: List[Resource] = links
        for i in range(len(links)):
            self.bows[i] = init_w
            self.boc_rngs.append(np.random.RandomState())
            self.bocs[i] = 0

        self.arr_itv_generator = IntervalGeneratorFactory.create(arr_type)
        self.pkts = []
        self.pkt_num = 0
        self.pkts_on_link = [[]]*len(links)
    
    def arrival_interval(self):
        return self.arr_itv_generator.get_itv()

    
    def run(self):
        self.env.process(self.generate_pkts())
        for i in range(len(self.links)):
            self.env.process(self.try_connecting(i))
        
    def generate_pkts(self):
        while True:
            yield self.env.timeout(self.arrival_interval())
            self.pkt_num += 1
            self.pkts.append(Pkt(self.id, self.env.now,))
            self.allocating()
            
    def allocating(self):
        if len(self.pkts) > 0:
            rv = random.uniform(0, 1)
            if rv < self.beta:
                pkt = self.pkts.pop(0)
                self.pkts_on_link[0].append(pkt)
            else:
                pkt = self.pkts.pop(0)
                self.pkts_on_link[1].append(pkt)
                        

                
    def try_connecting(self, linkid):
        while True:
            if self.links[linkid].count == 0:
                if len(self.pkts_on_link[linkid]) > 0:
                    if self.bocs[linkid] == 0:
                        with self.links[linkid].request() as req:
                            if not req.triggered:
                                yield self.env.timeout(self.col_time)
                            else:
                                yield self.env.timeout(self.suc_time)
                    else:
                        self.bocs[linkid] -= 1
                else:
                    self.bocs[linkid] = self.bocs[linkid] - 1 if self.bocs[linkid] > 0 else 0
                    
    
    def reset_bow(self, link_idx, flag = 0):
        if flag == 0:
            self.bows[link_idx] = self.init_w
        else:
            self.bows[link_idx] = min(self.bows[link_idx] * 2, self.max_w)
            
    def reset_boc(self, link_idx):
        self.bocs[link_idx] = self.boc_rngs[link_idx].randint(0, self.bows[link_idx])

