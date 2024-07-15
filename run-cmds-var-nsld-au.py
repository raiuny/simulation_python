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

def run(cmd, lam1, lam21,lam22, nmld, nsld1, nsld2, beta, tt, tf, mldW):
    log = f" --path ./log-au/log-var-W-{mldW}-{lam1:.4f}-{lam21:.4f}-{lam22:4f}-{nmld}-{nsld1}-{nsld2}-{beta:.3f}-{tt}-{tf}.csv --snum 5 --rt 20000000 --st 10000000"
    print(cmd + log)
    os.system(cmd+log)

lam210 = 0.1 / tt 
lam220 = 0.1 / tt 
lam1 = 0.1 / tt / 20
nmld = 20
nsld1 = 10
nsld2 = 10
with get_context("fork").Pool(16) as pool:
    for mldW in [16, 64, 128, 1024]:
        for nsld in [5, 10, 15, 20, 25, 30]:
            nsld1 = nsld2 = nsld
            lam21 = lam210 / nsld1
            lam22 = lam220 / nsld2
            cmd_run = cmd + f"--nsld1 {nsld1} --nsld2 {nsld2} --nmld {nmld} --lam1 {lam1} --lam21 {lam21}  --lam22 {lam22} --mldW1 {mldW} --beta {0.500}"
            pool.apply_async(run, (cmd_run, lam1, lam21, lam22, nmld, nsld1, nsld2, 0.5, tt, tf, mldW))
    pool.close()
    pool.join()
    