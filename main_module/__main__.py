"""
Main file for code library.

Author
------
::

    Author: Diptesh
    Date: Apr 17
    License: BSD 3-Clause
"""

# pylint: disable=invalid-name

# =============================================================================
# --- Import libraries
# =============================================================================

import time
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import lib.cfg
import lib.stat
import lib.opt

from lib import metrics
from lib import utils

# =============================================================================
# --- DO NOT CHANGE ANYTHING FROM HERE
# =============================================================================

__version__ = lib.cfg.__version__
__doc__ = lib.cfg.__doc__
path = lib.cfg.path + "data/"
module = lib.cfg.module

sep: str = "-" * 70
start: int = time.time_ns()
print(sep, "\n" + __doc__, "v" + __version__, "\n" + sep)

# =============================================================================
# --- Arguments
#
# cnt : int
# =============================================================================

CLI = argparse.ArgumentParser()
CLI.add_argument("-c", "--count",
                 nargs=1,
                 type=int,
                 default=[10],)

args = CLI.parse_args()

cnt = args.count[0]

# =============================================================================
# --- Main
# =============================================================================

if __name__ == '__main__':
    # ---- Clustering
    start = time.time_ns()
    fn_ip = "store.csv"
    df_ip = pd.read_csv(path + "input/" + fn_ip)
    opt_cluster = lib.stat.Cluster(x_var=["x1"], max_clus=5)
    df_op = opt_cluster.gap(df=df_ip)
    clus_stats = opt_cluster.gap_val
    clus_stats["Cluster"] = list(clus_stats.index + 1)
    print(utils.table_output(df=clus_stats,
                             col=["Cluster", "Gap", "SD"],
                             header=["Cluster",
                                     "Gap statistic",
                                     "Simulation error"]))
    print("Optimal cluster:", max(df_op["cluster"]))
    print(sep,
          utils.elapsed_time("Total time for clustering:", start),
          sep,
          sep="\n")
    # ---- Traveling salesman
    start = time.time_ns()
    df_ip = pd.read_csv(path + "input/us_city.csv")
    df_ip = df_ip.iloc[:10, :]
    tsp = lib.opt.TSP()
    opt = tsp.solve(loc=df_ip["city"].tolist(),
                    lat=df_ip["lat"].tolist(),
                    lon=df_ip["lng"].tolist(),
                    debug=False)
    df_op = pd.DataFrame(data=list(opt[2]), columns=["city", "dist"])
    df_op = pd.merge(df_op,
                     df_ip,
                     how="left",
                     on="city",
                     copy=False)
    df_op = df_op.append(df_op.iloc[0, :]).reset_index(drop=True)
    df_op.iloc[len(df_op) - 1, 1] = 0
    print(utils.table_output(df=df_op,
                             col=["city", "dist", "lat", "lng"],
                             header=["City", "Distance",
                                     "Latitude", "Longitude"],
                             precision=3))
    plt.plot(df_op["lat"], df_op["lng"], marker='o', color='b', zorder=1)
    plt.show()
    print(sep,
          utils.elapsed_time("Total time for TSP:", start),
          sep,
          sep="\n")
    # ---- Time series using XGBoost
    start = time.time_ns()
    n_per = 30
    df_ip = pd.read_csv(path + "input/data_ts_vars.csv")
    t_per = len(df_ip)
    df_pred = df_ip.iloc[t_per - n_per:, :]
    df_train = df_ip.iloc[0:t_per - n_per - 10, :]
    df_test = df_ip.iloc[t_per - n_per - 10:t_per - n_per, :]
    y_var = 'sales_q'
    x_var = ['sales_dollars', 'cost']
    xgb_mod = lib.stat.XGBoost(y_train=df_train[y_var],
                               x_train=df_train[x_var],
                               y_test=df_test[y_var],
                               x_test=df_test[x_var],
                               opts={"seed": 123456789,
                                     "n_jobs": -1,
                                     "k_fold": 10,
                                     "method": "time_series",
                                     "n_iter": 10})
    xgb_mod.fit()
    y_hat = xgb_mod.predict(df_ip[x_var])
    y_hat = [int(np.round(x)) for x in y_hat]
    df_ip.loc[:, "pred"] = y_hat
    df_ip[[y_var, "pred"]].plot()
    tmp = df_ip.dropna()
    xgb_op = pd.DataFrame(data=[["RSQ",
                                 metrics.rsq(tmp[y_var].tolist(),
                                             tmp["pred"].tolist())],
                                ["RMSE",
                                 metrics.rmse(tmp[y_var].tolist(),
                                              tmp["pred"].tolist())],
                                ["MSE",
                                 metrics.mse(tmp[y_var].tolist(),
                                             tmp["pred"].tolist())]],
                          columns=["Metric", "Value"])
    print(utils.table_output(df=xgb_op,
                             col=list(xgb_op.columns),
                             header=list(xgb_op.columns),
                             precision=3))
    print(sep,
          utils.elapsed_time("Total time for XGBoost:", start),
          sep,
          sep="\n")
