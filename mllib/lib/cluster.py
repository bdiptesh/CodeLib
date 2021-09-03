"""Clustering module."""

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

path = "/media/ph33r/Data/Project/mllib/dev/data/input/"

fn_ip = "store.csv"

x_var = ["x1"]
max_cluster = 10
# method = "one_se"
method = "gap_max"
nrefs = 10
seed = 1

df_ip = pd.read_csv(path + fn_ip)

df = df_ip[x_var]

max_cluster = min(max_cluster, len(df.drop_duplicates()))


def _nref(df):
    """Docstring."""
    x_cat = df.select_dtypes(include=['object', 'bool'])
    x_num = df.select_dtypes(include=['int', 'float64'])
    if not x_cat.empty:
        for i, c in enumerate(x_cat.columns):
            cat_val_list = df[c].unique()
            uniqu_val = len(cat_val_list)
            temp_cnt = 0
            while temp_cnt != uniqu_val:
                temp_d = np.random.choice(cat_val_list,
                                          size=len(df),
                                          p=[1.0/uniqu_val] * uniqu_val)
                temp_cnt = len(set(temp_d))
            temp_d = pd.DataFrame(temp_d)
            temp_d.columns = [c]
            if i == 0:
                x_cat_d = temp_d
            else:
                x_cat_d = x_cat_d.join(temp_d)
        df_sample = x_cat_d
    if not x_num.empty:
        for i, c in enumerate(x_num.columns):
            temp_d = np.random.uniform(low=min(df[c]),
                                       high=max(df[c]),
                                       size=len(df))
            temp_d = pd.DataFrame(temp_d)
            temp_d.columns = [c]
            if i == 0:
                x_cont_d = temp_d
            else:
                x_cont_d = x_cont_d.join(temp_d)
        if not x_cat.empty:
            df_sample = df_sample.join(x_cont_d)
        else:
            df_sample = x_cont_d
    df_sample = pd.get_dummies(data=df_sample, drop_first=True)
    df_sample = pd.DataFrame(scale(df_sample))
    return df_sample


df_clus = pd.get_dummies(data=df, drop_first=True)
df_clus_ip = pd.DataFrame(scale(df_clus))

gaps = np.zeros(max_cluster)
sks = np.zeros(max_cluster)

df_result = pd.DataFrame({"cluster": [], "gap": [], "sk": []})

for gap_index, k in enumerate(range(1, max_cluster+1)):
    # Holder for reference dispersion results
    ref_disps = np.zeros(nrefs)
    # For n references, generate random sample and perform kmeans getting
    # resulting dispersion of each loop
    for i in range(nrefs):
        # Create new random reference set
        random_ref = _nref(df)
        # Fit to it
        km = KMeans(k, random_state=seed)
        km.fit(random_ref)
        ref_disp = km.inertia_
        ref_disps[i] = ref_disp
    # Fit cluster to original data and create dispersion
    km = KMeans(k, random_state=seed)
    km.fit(df_clus_ip)
    orig_disp = km.inertia_
    # Calculate gap statistic
    if orig_disp > 0.0:
        gap = np.log(np.mean(ref_disps)) - np.log(orig_disp)
    else:
        gap = np.inf
    # Standard error
    if sum(ref_disps) == 0.0:
        sk = 0.0
    else:
        sdk = np.std(np.log(ref_disps))
        sk = sdk * np.sqrt(1.0 + 1.0 / nrefs)
    # Assign this loop's gap statistic and sk to gaps and sks
    gaps[gap_index] = gap
    sks[gap_index] = sk
    # One SE
    if method == "one_se":
        if k > 1 and gaps[gap_index-1] >= gap - sk:
            opt_k = k-1
            km = KMeans(opt_k, random_state=seed)
            km.fit(df_clus_ip)
            clus_op = km.labels_
    df_result = df_result.append({"cluster": k,
                                  "gap": gap,
                                  "sk": sk},
                                 ignore_index=True)
    if method == "gap_max":
        opt_k = np.argmax(gaps) + 1
        km = KMeans(opt_k, random_state=seed)
        km.fit(df_clus_ip)
        clus_op = km.labels_

df_result
