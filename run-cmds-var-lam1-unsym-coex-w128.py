from multiprocessing import Pool
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
lam21 = 0.001
lam22 = 0.002
def run(cmd, lam1, lam21,lam22, nmld, nsld1, nsld2, beta, tt, tf):
    path = f" --path ./log/log-unsym-w128-{lam1:.4f}-{lam21:.4f}-{lam22:4f}-{nmld}-{nsld1}-{nsld2}-{beta:.3f}-{tt}-{tf}.csv --snum 10"
    print(cmd + path)
    os.system(cmd + path)

with Pool(8) as pool:
    for lam1 in np.arange(0.0002, 0.0034, 0.0002):
        cmd_run = cmd + f"--nsld1 {nsld1} --nsld2 {nsld2} --nmld {nmld} --lam1 {lam1} --lam21 {lam21} --lam22 {lam22} --beta {0.500} --mldW1 128 --rt 1000000"
        pool.apply_async(run, (cmd_run, lam1, lam21,lam22, nmld, nsld1, nsld2, 0.5, tt, tf))
    pool.close()
    pool.join()
    