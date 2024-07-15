from multiprocessing import Pool, get_context
import numpy as np
import os
cmd = "python simulation.py "

tt = 32
tf = 27
nmld = 0

def run(cmd, lam1, lam21,lam22, nmld, nsld1, nsld2, beta, tt, tf):
    log = f" --path ./log-1-20-0.2-0.1/log-var-{lam1:.4f}-{lam21:.4f}-{lam22:4f}-{nmld}-{nsld1}-{nsld2}-{beta:.3f}-{tt}-{tf}.csv --snum 5 --rt 20000000 --st 10000000"
    print(cmd + log)
    os.system(cmd+log)

lam210 = 0.1 / tt 
lam220 = 0.1 / tt 
lam1 = 0.1 / tt 
nmld = 1
nsld1 = 10
nsld2 = 10
with get_context("fork").Pool(20) as pool:
    for lam2 in np.arange(0.1, 1.1, 0.1):
        for nsld in [5, 10, 16]:
            nsld1 = nsld
            nsld2 = 20 - nsld1 
            lam21 = lam2 / 20 / tt
            lam22 = lam21
            cmd_run = cmd + f"--nsld1 {nsld1} --nsld2 {nsld2} --nmld {nmld} --lam1 {lam1} --lam21 {lam21}  --lam22 {lam22} --beta {0.500}"
            pool.apply_async(run, (cmd_run, lam1, lam21, lam22, nmld, nsld1, nsld2, 0.5, tt, tf))
    pool.close()
    pool.join()