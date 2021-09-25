"""
Test suite module for ``knn``.

Credits
-------
::

    Authors:
        - Diptesh
        - Madhu

    Date: Sep 25, 2021
"""

# pylint: disable=invalid-name
# pylint: disable=wrong-import-position

import unittest
import warnings
import re
import sys

from inspect import getsourcefile
from os.path import abspath

import pandas as pd

from sklearn.model_selection import train_test_split as split

# Set base path
path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+)(\/tests.*)", "\\1", path)

sys.path.insert(0, path)

from mllib.lib.knn import KNN  # noqa: F841

# =============================================================================
# --- DO NOT CHANGE ANYTHING FROM HERE
# =============================================================================

path = path + "/data/input/"

# =============================================================================
# --- User defined functions
# =============================================================================


def ignore_warnings(test_func):
    """Suppress deprecation warnings."""

    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)
    return do_test


class Test_Knn(unittest.TestCase):
    """Test suite for module ``KNN``."""

    def setUp(self):
        """Set up for module ``KNN``."""

    def test_knn_class(self):
        """KNN: Test for classification."""
        df_ip = pd.read_csv(path + "iris.csv")
        df_ip = df_ip[["y", "x1", "x2"]]
        df_train, df_test = split(df_ip,
                                  stratify=df_ip["y"],
                                  test_size=0.1,
                                  random_state=42)
        mod = KNN(df_train, "y", ["x1", "x2"], method="classify")
        y_hat = mod.predict(df_test[["x1", "x2"]]).tolist()
        y = df_test["y"].values.tolist()
        acc = round(len([i for i, j in zip(y, y_hat) if i == j]) / len(y), 2)
        self.assertGreaterEqual(acc, 0.93)

    def test_knn_reg(self):
        """KNN: Test for regression."""
        df_ip = pd.read_csv(path + "iris.csv")
        df_ip = df_ip[["y", "x1", "x2"]]
        df_train, df_test = split(df_ip,
                                  stratify=df_ip["y"],
                                  test_size=0.1,
                                  random_state=42)
        mod = KNN(df_train, "y", ["x1", "x2"], method="regression")
        y_hat = mod.predict(df_test[["x1", "x2"]]).tolist()
        y = df_test["y"].values.tolist()
        acc = round(len([i for i, j in zip(y, y_hat) if i == j]) / len(y), 2)
        self.assertGreaterEqual(acc, 0.87)


# =============================================================================
# --- Main
# =============================================================================

if __name__ == '__main__':
    unittest.main()
