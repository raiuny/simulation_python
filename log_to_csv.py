import pandas as pd
import re
import numpy as np

data = pd.read_excel("result.xlsx")

def get_result(file):
    with open(file) as f:
        # beta = float(file.split("-")[-4])
        # print(beta)
        text = f.read()
        pattern = r"weighted e2e delay of mld: (\d+\.\d+)"
        matches_e2e_delay = float(re.findall(pattern, text)[0])
        pattern_mld = r"Throughput:\s*\n\(MLD\)\n([\d.]+\s+[\d.]+)"
        matches_thpt_mld = re.findall(pattern_mld, text)[0]
        thpts = np.array(str.split(matches_thpt_mld, '\t'), dtype=float)
        return matches_e2e_delay, thpts[0], thpts[1]
if __name__ == "__main__":
    # get_result("log/log-0.001-0.0002-20-0-20-0.100-32-27-W128.txt")
    log_path = "log2-var-beta-w128/"
    logfile = log_path + "log-%.3f-%.4f-%d-%d-%d-%.3f-%d-%d-W128.txt"
    lam1 = 0.001
    lam2 = 0.0002
    nmld = 20
    nsld1 = 0
    nsld2 = 20
    beta = 0.1
    tt = 32
    tf = 27
    # Wmld = 128
    e2e_delay_mld_res = {}
    thpt1_mld_res = {}
    thpt2_mld_res = {}
    df = {}
    for nsld1 in [0, 4, 10]:
        nsld2 = 20 - nsld1
        e2e_delay_mld_res[nsld1] = []
        thpt1_mld_res[nsld1] = []
        thpt2_mld_res[nsld1] = []
        df[nsld1] = {"beta":np.arange(0.1, 1.0, 0.01)}
        for beta in np.arange(0.1, 1.0, 0.01):
            file = logfile%(lam1, lam2, nmld, nsld1, nsld2, beta, tt, tf)
            e2e_delay_mld, thpt1_mld, thpt2_mld = get_result(file)
            e2e_delay_mld_res[nsld1].append(e2e_delay_mld)
            thpt1_mld_res[nsld1].append(thpt1_mld)
            thpt2_mld_res[nsld1].append(thpt2_mld)
    for k, v in e2e_delay_mld_res.items():
        df[k]["E2E Delay(python sim)"] = v
    for k, v in thpt1_mld_res.items():
        df[k]["throughput of MLD on Link1(python sim)"] = v
    for k, v in thpt2_mld_res.items():
        df[k]["throughput of MLD on Link2(python sim)"] = v
    for k, d in df.items():
        pd.DataFrame(d).to_csv(f"result/var_beta_sld_{k}_{20-k}_W128.csv", index=None)