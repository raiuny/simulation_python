import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline
lam1 = 0.002
lam21 = 0.001
lam22 = 0.001
nmld = 10
tt = 32
tf = 27

def plot(x, y, label):
    xp, yp = [], []
    for xi, yi in zip(x, y):
        if yi < 200:
            xp.append(xi)
            yp.append(yi)
    plt.plot(xp, yp, label=label)
best_beta = []
for nsld1 in np.arange(0, 21, 1):
    nsld2 = 20 - nsld1
    df = None
    for beta in np.arange(0, 1.01, 0.01):
        csvpath = f"./log/log-var-beta-{lam1:.4f}-{lam21:.4f}-{lam22:4f}-{nmld}-{nsld1}-{nsld2}-{beta:.3f}-{tt}-{tf}.csv"
        tmp = pd.read_csv(csvpath).drop(columns=["Unnamed: 0"]).mean().T
        df = pd.concat([df, tmp], axis=1)
    plot(np.arange(0, 1.01, 0.01), df.T['weighted e2e delay of mld'], label = f"{nsld1}:{nsld2}")
    df.to_csv(f"./log/sumup-var-beta-{lam1:.4f}-{lam21:.4f}-{lam22:4f}-{nmld}-{nsld1}-{nsld2}-{beta:.3f}-{tt}-{tf}.csv")
    idx = np.argmin(df.T["weighted e2e delay of mld"])
    best_beta.append(np.arange(0, 1.01, 0.01)[idx])
print(best_beta)
plt.legend()
plt.grid()
plt.show()