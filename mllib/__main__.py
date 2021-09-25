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
from lib.model import GLMNet  # noqa: F841

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
fn_ip = "store.csv"

# =============================================================================
# --- Main
# =============================================================================

if __name__ == '__main__':
    start = time.time_ns()
    # --- Clustering
    start_t = time.time_ns()
    df_ip = pd.read_csv(path + "input/" + fn_ip)
    clus_sol = Cluster(df=df_ip, x_var=["x1"])
    clus_sol.opt_k()
    print("Clustering\n",
          "optimal k = " + str(clus_sol.optimal_k),
          elapsed_time("Time", start_t),
          sep="\n")
    # --- GLMNet
    start_t = time.time_ns()
    df_ip = pd.read_csv(path + "input/test_glmnet.csv")
    glm_mod = GLMNet(df=df_ip,
                     y_var=["y"],
                     x_var=["x1", "x3"])
    print("\nGLMNet\n")
    for k, v in glm_mod.model_summary.items():
        print(k, str(v).rjust(69 - len(k)))
    print(elapsed_time("Time", start_t),
          sep="\n")
    # --- EOF
    print(sep, elapsed_time("Total time", start), sep, sep="\n")
