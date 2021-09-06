"""
Machine Learning Library.

Objective:
    - Illustrate module APIs with some examples.

Credits
-------
::

    Authors:
        - Diptesh

    Date: Sep 01, 2021
"""

# pylint: disable=invalid-name

# =============================================================================
# --- Import libraries
# =============================================================================

import argparse
import time

import pandas as pd

from lib import cfg, utils  # noqa: F841
from lib.cluster import Cluster  # noqa: F841
from lib.glmnet import GLMNet  # noqa: F841

# =============================================================================
# --- DO NOT CHANGE ANYTHING FROM HERE
# =============================================================================

__version__ = cfg.__version__
__doc__ = cfg.__doc__
path = cfg.path + "data/"
elapsed_time = utils.elapsed_time

sep = "-" * 70
print(sep, "\n" + __doc__, "v" + __version__, "\n" + sep + "\n")

# =============================================================================
# --- Arguments
#
# filename: str
# =============================================================================

CLI = argparse.ArgumentParser()

CLI.add_argument("-f", "--filename",
                 nargs=1,
                 type=str,
                 default=["store.csv"],
                 help="input csv filename")

args = CLI.parse_args()

fn_ip = args.filename[0]

# =============================================================================
# --- Main
# =============================================================================

if __name__ == '__main__':
    start = time.time_ns()
    # --- Clustering
    df_ip = pd.read_csv(path + "input/" + fn_ip)
    clus_sol = Cluster(df=df_ip, x_var=["x1"])
    clus_sol.opt_k()
    print("Clustering\n",
          "optimal k = " + str(clus_sol.optimal_k),
          elapsed_time("Time", start),
          sep="\n")
    # --- GLMNet
    df_ip = pd.read_csv(path + "input/test_glmnet.csv")
    glm_mod = GLMNet(df=df_ip,
                     y_var=["y"],
                     x_var=["x1", "x2", "x3"])
