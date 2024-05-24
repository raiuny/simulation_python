import pandas as pd
import re
import numpy as np

data = pd.read_excel("result.xlsx")

def get_result(file):
    with open(file) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 3:
                mld_thpts = np.sum(np.array(line.strip().split('\t'), dtype=float))
            if i == 5:
                sld_thpts = np.sum(np.array(line.strip().split('\t'), dtype=float))
            if i == 10:
                mld_ac_delay = np.mean(np.array(line.strip().split('\t'), dtype=float))
            if i == 16:
                sld_ac_delay = np.mean(np.array(line.strip().split('\t'), dtype=float))
            if i == 12:
                mld_queueing_delay = np.mean(np.array(line.strip().split('\t'), dtype=float))
            if i == 18:
                sld_queueing_delay = np.mean(np.array(line.strip().split('\t'), dtype=float))
            if i == 22:
                mld_p = np.mean(np.array(line.strip().split('\t'), dtype=float))
            if i == 24:
                sld_p = np.mean(np.array(line.strip().split('\t'), dtype=float))
        total_thpt = mld_thpts + sld_thpts
        ac_delay = mld_ac_delay
        queueing_delay = mld_queueing_delay
        # print(total_thpt, ac_delay, queueing_delay, ac_delay + queueing_delay)
        return total_thpt, ac_delay, queueing_delay, ac_delay + queueing_delay, mld_p
        
            
        
if __name__ == "__main__":
    # get_result("log/log-0.001-0.0002-20-0-20-0.100-32-27-W128.txt")
    log_path = "log-sym-v1/"
    logfile = log_path + "log-%.4f-%.4f-%d-%d-%d-%.3f-%d-%d.txt"
    lam1 = 0.001
    lam2 = 0.0002
    nmld = 20
    nsld1 = 0
    nsld2 = 20
    beta = 0.1
    tt = 32
    tf = 27
    # Wmld = 128
    df = {}
    df["throughput"] = []
    df["access_delay"] = []
    df["queueing_delay"] = []
    df["e2e_delay"] = []
    df["p"] = []
    for lam1 in np.arange(0.0002, 0.0032, 0.0002):
        file = logfile%(lam1,lam1/2, 10, 10, 10, 0.5, 32,27)
        thpt, ad, qd, ed, p = get_result(file)
        df["throughput"].append(thpt)
        df["access_delay"].append(ad)
        df["queueing_delay"].append(qd)
        df["e2e_delay"].append(ed)
        df["p"].append(p)
        
    # for k, v in e2e_delay_mld_res.items():
    #     df[k]["E2E Delay(python sim)"] = v
    # for k, v in thpt1_mld_res.items():
    #     df[k]["throughput of MLD on Link1(python sim)"] = v
    # for k, v in thpt2_mld_res.items():
    #     df[k]["throughput of MLD on Link2(python sim)"] = v
    # for k, d in df.items():
    #     pd.DataFrame(d).to_csv(f"result/var_beta_sld_{k}_{20-k}_W128.csv", index=None)
    pd.DataFrame(df).to_csv(f"var_lambda_sym.csv", index=None)