from multiprocessing import Pool, get_context
import numpy as np
import os
cmd = "python simulation.py "

# sld1 0.0002
# sld2 0.0002
# mld  0.0002
tt = 32
tf = 27
nsld1 = 10
nsld2 = 10
nmld = 10
lam1 = 0.002
lam21 = 0.0002
lam22 = 0.0002
def run(cmd, lam1, lam21,lam22, nmld, nsld1, nsld2, beta, tt, tf):
    path = f" --path ./log2/log-var-beta-{lam1:.4f}-{lam21:.4f}-{lam22:4f}-{nmld}-{nsld1}-{nsld2}-{beta:.3f}-{tt}-{tf}.csv --snum 4"
    print(cmd + path)
    os.system(cmd + path)

with get_context("fork").Pool(8) as pool:
    for nsld1 in range(0, 21, 1):
        nsld2 = 20 - nsld1
        for beta in np.arange(0.40, 0.61, 0.01):
            cmd_run = cmd + f"--nsld1 {nsld1} --nsld2 {nsld2} --nmld {nmld} --lam1 {lam1} --lam21 {lam21} --lam22 {lam22} --beta {beta} --mldW1 16 --rt 1000000"
            pool.apply_async(run, (cmd_run, lam1, lam21,lam22, nmld, nsld1, nsld2, beta, tt, tf))
    pool.close()
    pool.join()
    