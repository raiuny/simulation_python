# M/M/C/K queue system
import simpy
import random
from time import time
import numpy as np
from typing import List
from math import factorial

from simpy.resources.resource import Release, Request
from packet import Pkt
from mld import MLD
from arrival_model import ArrivalType

class Params(object):
    arrival_rate = 0.01 # per node per slot
    nlink = 2 # 2 links
    nmld = 20
    nsld = 0
    W = 16
    K = 6 
    sim_duration = 1e6
    num_runs = 10
    wait_time = []
    service_time = []
    fin_counter = 0
    collide_counter = 0
    queue_len_time = 0
    tt = 36
    tf = 28

class LinkResource(simpy.Resource):
    def __init__(self, env: simpy.Environment, capacity: int = 1):
        super().__init__(env, capacity)
        self.cnt = 0
    
    def request(self) -> Request:
        self.cnt += 1
        if self.cnt > 1:
            return False
        return super().request()

    def release(self, request: Request) -> Release:
        
        return super().release(request)
    
    
        
class System(object):
    def __init__(self):
        self.env = simpy.Environment()
        
        # MLD
        self.mlds: List[MLD] = []
        self.links: List[simpy.Resource] = []
        for i in range(Params.nlink):
            self.links.append(simpy.Resource(self.env, capacity=1)) # 一般资源 先进先出
        for i in range(Params.nmld):
            self.mlds.append(MLD(i, self.env, self.links, ArrivalType.BERNOULLI, Params.W, Params.K, Params.tt, Params.tf))
        
        self.Pkt_counter = 0
        self.PktList: List[Pkt] = []
        
        self.fin_counter = simpy.Resource(self.env, capacity=1)
        self.queue_len = 0
        # 优先资源: PriorityResource 一般资源: Resource (FCFS)
    
    def serving(self, pkt: Pkt):
        with self.links.request() as req:
            yield req
            pkt.ser_time = self.env.now
            Params.wait_time.append(pkt.ser_time - pkt.arr_time)
            service_time = Params.tt
            yield self.env.timeout(service_time)
            pkt.fin_time = self.env.now
            self.queue_len -= 1
            Params.service_time.append(pkt.fin_time - pkt.ser_time)
            Params.fin_counter += 1
    
    def colliding(self, pkt: Pkt):
        collision_time = Params.tf
        yield self.env.timeout(collision_time)
        Params.collide_counter += 1
            
    def run(self):
        for mld in self.mlds:
            self.env.process(mld.run())
        
        self.env.process(self.step_process())
        self.env.run(until=Params.sim_duration)
        return np.mean(Params.wait_time), np.var(Params.wait_time), np.mean(Params.service_time), np.var(Params.service_time)
    
    def calc_rho(self):
        busy_time = 0
        for c in self.customerList:
            if c.ser_time is not None:
                if c.fin_time is not None:
                    busy_time += c.fin_time - c.ser_time
                else:
                    busy_time += Params.sim_duration - c.ser_time               
        return busy_time / Params.sim_duration
    
    def step_process(self):
        while True:
            Params.queue_len_time += max(self.queue_len - Params.nserver, 0)
            yield self.env.timeout(1)

if __name__ == "__main__":
    sys = System()
    W_q, Var_W_q, serving_time, Var_serving_time= sys.run()
    served_num = Params.fin_counter 
    rho = sys.calc_rho()
    
    print(f"Simulation:")
    print(f"W_q: {W_q}, mean_serving_time: {serving_time}, N_q: {Params.queue_len_time / Params.sim_duration}, rho: {rho}")
    print(f"Var(W_q): {Var_W_q}, Var(serving_time): {Var_serving_time}")
    print(f"Formula:")
    if Params.nserver == 1: # formula of M/M/1 queue system 
        rho_formula = Params.arrival_rate / Params.service_rate
        print(f"W_q: {rho_formula / (1 - rho_formula) / Params.service_rate}, N_q: {rho_formula**2 / (1 - rho_formula)}, rho: {rho_formula}, mean_serving_time: {1 / Params.service_rate}") 
        print(f"Var(W_q): {1 / (Params.service_rate - Params.arrival_rate) ** 2 - 1 / Params.service_rate ** 2}, Var(serving_time): {1 / Params.service_rate ** 2}")
    else:
        if Params.maxLen_queue == np.inf:
            # formula of M/M/c/∞ queue system
            c = Params.nserver
            rho_formula = Params.arrival_rate / (c * Params.service_rate)
            Erlang_C = 1 / (1 + (1 - rho_formula) * (factorial(c)/(c*rho_formula)**c * sum([(c * rho_formula) ** i / factorial(i) for i in range(c)])))
            N_q = rho_formula / (1 - rho_formula) * Erlang_C
            W_q = Erlang_C / (c * Params.service_rate - Params.arrival_rate)
            print(f"W_q: {W_q}, N_q: {N_q}, rho: {rho_formula * c}, mean_serving_time: {1 / Params.service_rate}") 
        else:
            c = Params.nserver
            K = Params.maxLen_queue
            assert K > c
            rho =  Params.arrival_rate / Params.service_rate
            pi_0 = 1 / (sum([ rho ** i / factorial(i) for i in range(c+1)]) + rho ** c / factorial(c) * sum([ rho ** (i+1)/ c ** (i+1) for i in range(K-c)]))
            rho_c = rho / c
            W_q = pi_0 * rho_c * rho ** c / (Params.arrival_rate * (1 - rho_c) ** 2 * factorial(c))
            N_q = W_q * Params.arrival_rate # Little's Formula
            print(f"W_q: {W_q}, N_q: {N_q}, rho: {rho}, mean_serving_time: {1 / Params.service_rate}") 