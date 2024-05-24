from multiprocessing import Pool, get_context
import numpy as np
import os
cmd = "python simulation.py "

# sld1 0.0002
# sld2 0.0002
# mld  0.0002
tt = 32
tf = 27
nmld = 0

def run(cmd, lam1, lam21,lam22, nmld, nsld1, nsld2, beta, tt, tf):
    log = f" --path ./log-n/log-var-nsld1-{lam1:.4f}-{lam21:.4f}-{lam22:4f}-{nmld}-{nsld1}-{nsld2}-{beta:.3f}-{tt}-{tf}.csv --snum 2"
    print(cmd + log)
    # os.system(cmd+log)

lam21 = 0.005
lam22 = 0 
lam1 = 0
nsld2 = 0

for nsld1 in np.arange(10, 110, 10):
    cmd_run = cmd + f"--nsld1 {nsld1} --nsld2 {nsld2} --nmld {nmld} --lam1 {lam1} --lam21 {lam21}  --lam22 {0} --beta {0.500}"
    run(cmd_run, lam1, lam21, lam22, nmld, nsld1, nsld2, 0.5, tt, tf)
    