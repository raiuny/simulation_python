from multiprocessing import Pool, get_context
import numpy as np
import os
cmd = "python simulation.py "

tt = 32
tf = 27
nmld = 0

def run(cmd, lam1, lam21,lam22, nmld, nsld1, nsld2, beta, tt, tf, sldW):
    log = f" --path ./log-var-lambda-sld/log-var-sldW-{sldW}-{lam1:.4f}-{lam21:.4f}-{lam22:4f}-{nmld}-{nsld1}-{nsld2}-{beta:.3f}-{tt}-{tf}.csv --snum 5 --rt 20000000 --st 10000000"
    print(cmd + log)
    os.system(cmd+log)

lam210 = 0.1 / tt 
lam220 = 0.1 / tt 
lam1 = 0.1 / tt / 20
nmld = 0
nsld1 = 20
nsld2 = 0
with get_context("fork").Pool(20) as pool:
    for sldW in [16, 64, 128, 1024]:
        for lam210 in np.arange(0.1, 1.1, 0.1):
            lam21 = lam210 / nsld1 / tt
            lam22 = 0
            cmd_run = cmd + f"--nsld1 {nsld1} --nsld2 {nsld2} --nmld {nmld} --lam1 {lam1} --lam21 {lam21}  --lam22 {lam22} --sldW {sldW} --beta {0.500}"
            pool.apply_async(run, (cmd_run, lam1, lam21, lam22, nmld, nsld1, nsld2, 0.5, tt, tf, sldW))
    pool.close()
    pool.join()
    