from multiprocessing import Pool
import numpy as np
import os
cmd = "python simulation.py "

# sld1 0.0002
# sld2 0.0002
# mld  0.0004
tt = 32
tf = 27
nsld_total = 20
nmld = 20
lam1 = 0.002
lam2 = 0.0001

def run(cmd, lam1, lam2, nmld, nsld1, nsld2, beta, tt, tf):
    log = f" > ./log/log-{lam1}-{lam2}-{nmld}-{nsld1}-{nsld2}-{beta:.3f}-{tt}-{tf}.txt 2>&1"
    print(cmd + log)
    os.system(cmd+log)

with Pool(16) as pool:
    for nsld1 in [0, 4, 10]:
        nsld2 = nsld_total - nsld1
        for beta in np.arange(0.1, 1.0, 0.01):
            cmd_run = cmd + f"--nsld1 {nsld1} --nsld2 {nsld2} --nmld {nmld} --lam1 0.0002 --lam2 0.0002 --beta {beta:.3f}"
            pool.apply_async(run, (cmd_run, lam1, lam2, nmld, nsld1, nsld2, beta, tt, tf))
    pool.close()
    pool.join()
    