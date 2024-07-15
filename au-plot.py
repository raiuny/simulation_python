import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


mldW = 16
tt = 32
tf = 27
nmld = 20
lam1 = 0.1 / tt / nmld
lam210 = 0.1 / tt
lam220 = 0.1 / tt
beta = 0.5
w_range = [16, 64, 128, 1024]
acc_delay_sim = {}
acc_delay_model = {}
q_delay_sim = {}
q_delay_model = {}
for w in w_range:
    acc_delay_sim[w] = []
    q_delay_sim[w] = []
    for nsld in np.arange(5, 35, 5):
        nsld1 = nsld2 = nsld
        lam21 = lam210 / nsld1
        lam22 = lam220 / nsld2
        file = f"./log-au/log-var-W-{w}-{lam1:.4f}-{lam21:.4f}-{lam22:4f}-{nmld}-{nsld1}-{nsld2}-{beta:.3f}-{tt}-{tf}.csv"
        data = pd.read_csv(file)
        acd1 = data["Access Delay of MLD on Link 1"].mean()
        acd1_std = data["Access Delay of MLD on Link 1"].std() 
        acd2 = data["Access Delay of MLD on Link 2"].mean()
        acd2_std = data["Access Delay of MLD on Link 2"].std()
        qd1 = data["Queueing Delay of MLD on Link 1"].mean()
        qd1_std = data["Queueing Delay of MLD on Link 1"].std()
        qd2 = data["Queueing Delay of MLD on Link 2"].mean()
        qd2_std = data["Queueing Delay of MLD on Link 2"].std()
        print(acd1, acd2, qd1, qd2, acd1_std, acd2_std, qd1_std, qd2_std)
        if acd1_std < acd2_std:
            acc_delay_sim[w].append(acd1)
        else:
            acc_delay_sim[w].append(acd2)
        if qd1_std < qd2_std:
            q_delay_sim[w].append(qd1)
        else:
            q_delay_sim[w].append(qd2)

plt.figure(1)
for w in w_range:
    plt.plot(np.arange(5,35,5), acc_delay_sim[w], label = f"W={w} sim")
    plt.scatter(np.arange(5,35,5), acc_delay_sim[w])
plt.grid()
plt.legend()
plt.ylabel("access delay")
plt.xlabel("nsld")
plt.savefig("all_unsat_acd_vs_nsld.png")

plt.figure(2)
for w in w_range:
    plt.plot(np.arange(5,35,5), q_delay_sim[w], label = f"W={w} sim")
    plt.scatter(np.arange(5,35,5), q_delay_sim[w])
plt.grid()
plt.legend()
plt.ylabel("queueing delay")
plt.xlabel("nsld")
plt.savefig("all_unsat_qd_vs_nsld.png")