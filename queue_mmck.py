# M/M/C/K queue system
import simpy
import random
from time import time
import numpy as np
from typing import List
from math import factorial
class Params(object):
    arrival_rate = 0.8
    service_rate = 0.5
    nserver = 3 # C
    maxLen_queue = 16 # K
    sim_duration = 1e6
    num_runs = 10
    wait_time = []
    service_time = []
    fin_counter = 0
    queue_len_time = 0
    
class Customer(object):
    def __init__(self, pid, arr_time):
        self.pid = pid
        self.arr_time = arr_time
        self.ser_time = None
        self.fin_time = None
    
class System(object):
    def __init__(self):
        self.env = simpy.Environment()
        self.customer_counter = 0
        self.customerList: List[Customer] = []
        self.server = simpy.Resource(self.env, capacity=Params.nserver) # 一般资源 先进先出
        self.fin_counter = simpy.Resource(self.env, capacity=1)
        self.queue_len = 0
        # 优先资源: PriorityResource 一般资源: Resource (FCFS)

    def generate_customers(self):
        while True:
            customer = Customer(self.customer_counter, self.env.now)
            if self.queue_len + 1 <= Params.maxLen_queue:
                self.customer_counter += 1
                self.queue_len += 1
                self.customerList.append(customer)
                self.env.process(self.serving(self.customerList[self.customer_counter-1]))
            arrival_interval = random.expovariate(Params.arrival_rate)
            yield self.env.timeout(arrival_interval)
    
    def serving(self, cst: Customer):
        with self.server.request() as req:
            yield req
            cst.ser_time = self.env.now
            Params.wait_time.append(cst.ser_time - cst.arr_time)
            service_time = random.expovariate(Params.service_rate)
            yield self.env.timeout(service_time)
            cst.fin_time = self.env.now
            self.queue_len -= 1
            Params.service_time.append(cst.fin_time - cst.ser_time)
            Params.fin_counter += 1
            
    def run(self):
        self.env.process(self.generate_customers())
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