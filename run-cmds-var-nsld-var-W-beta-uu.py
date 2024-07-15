from multiprocessing import Pool, get_context
import numpy as np
import os
cmd = "python simulation.py "

tt = 32
tf = 27
nmld = 0

def run(cmd, lam1, lam21,lam22, nmld, nsld1, nsld2, beta, tt, tf, w):
    log = f" --path ./log-var-nsld-var-W-beta-uu/log-W-{w}-{lam1:.4f}-{lam21:.4f}-{lam22:4f}-{nmld}-{nsld1}-{nsld2}-{beta:.3f}-{tt}-{tf}.csv --snum 5 --rt 20000000 --st 10000000"
    print(cmd + log)
    os.system(cmd+log)

lam2 = 0.5 / tt
lam1 = 0.1 / tt 
nmld = 1
with get_context("fork").Pool(20) as pool:
    for w in [1, 2, 4, 8, 16, 32]:
        for nsld in [5, 10, 16]:
            for beta in np.arange(0.1, 1.0, 0.1):
                nsld1 = nsld
                nsld2 = 20 - nsld1 
                lam21 = lam2 / 20
                lam22 = lam21
                cmd_run = cmd + f"--nsld1 {nsld1} --nsld2 {nsld2} --nmld {nmld} --lam1 {lam1} --lam21 {lam21}  --lam22 {lam22} --mldW1 {w} --mldW2 {w} --beta {beta}"
                pool.apply_async(run, (cmd_run, lam1, lam21, lam22, nmld, nsld1, nsld2, beta, tt, tf, w))
    pool.close()
    pool.join()