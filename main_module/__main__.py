"""
Main file for code library.

Author
------
::

    Author: Diptesh
    Date: Jun 14, 2019
    License: BSD 3-Clause
"""

# pylint: disable=invalid-name

# =============================================================================
# --- Import libraries
# =============================================================================

import time
import argparse

import pandas as pd
import matplotlib.pyplot as plt

import lib.cfg
import lib.stat
import lib.opt

from lib import utils

# =============================================================================
# --- DO NOT CHANGE ANYTHING FROM HERE
# =============================================================================

__version__ = lib.cfg.__version__
__doc__ = lib.cfg.__doc__
path = lib.cfg.path + "data/"
module = lib.cfg.module

sep = "-" * 70
start = time.time_ns()
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
# --- User defined functions
# =============================================================================


def t_sum(x, y):
    """Sum 2 numbers.

    Parameters
    ----------
    :x: int

    :y: int

    """
    return x + y


# =============================================================================
# -- Main
# =============================================================================

if __name__ == '__main__':
    # Clustering
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
    # Traveling salesman
    start = time.time_ns()
    df_ip = pd.read_csv(path + "input/us_city.csv")
    df_ip = df_ip.iloc[:10, :]
    tsp = lib.opt.TSP()
    opt = tsp.solve(loc=df_ip["city"].tolist(),
                    x=df_ip["lat"].tolist(),
                    y=df_ip["lng"].tolist(),
                    debug=0)
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
    print(sep,
          utils.elapsed_time("Total time for TSP:", start),
          sep,
          sep="\n")
    plt.show()
