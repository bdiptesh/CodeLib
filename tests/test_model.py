"""
Test suite module for ``model``.

Credits
-------
::

    Authors:
        - Diptesh

    Date: Sep 07, 2021
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
import numpy as np

# Set base path
path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+)(\/tests.*)", "\\1", path)

sys.path.insert(0, path)

from mllib.lib.model import create_lag_vars  # noqa: F841
from mllib.lib.model import GLMNet  # noqa: F841

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


class TestCreateLagVars(unittest.TestCase):
    """Test suite for UDF ``create_lag_vars``."""

    def setUp(self):
        """Set up for UDF ``create_lag_vars``."""

    def test_no_interval_specified(self):
        """Lag vars: Test when no interval is specified."""
        df_ip = pd.read_csv(path + "test_lag_var.csv")
        df_op = create_lag_vars(df=df_ip,
                                y_var=["y"],
                                x_var=["x1", "x2"])
        exp_op = df_ip[list(df_ip.columns[1:])].dropna().reset_index(drop=True)
        self.assertEqual(df_op.equals(exp_op), True)

    def test_interval_specified(self):
        """Lag vars: Test when interval is specified."""
        df_ip = pd.read_csv(path + "test_lag_var.csv")
        df_op = create_lag_vars(df=df_ip,
                                y_var=["y"],
                                x_var=["x1", "x2"],
                                n_interval="week")
        exp_op = df_ip[list(df_ip.columns[1:])].dropna().reset_index(drop=True)
        self.assertEqual(df_op.equals(exp_op), True)


class TestGLMNet(unittest.TestCase):
    """Test suite for module ``GLMNet``."""

    def setUp(self):
        """Set up for module ``GLMNet``."""

    def test_known_equation(self):
        """GLMNet: Test a known equation."""
        df_ip = pd.read_csv(path + "test_glmnet.csv")
        mod = GLMNet(df=df_ip,
                     y_var=["y"],
                     x_var=["x1", "x2", "x3"])
        op = mod.opt
        self.assertEqual(np.round(op.get('intercept'), 0), 100.0)
        self.assertEqual(np.round(op.get('coef')[0], 0), 2.0)
        self.assertEqual(np.round(op.get('coef')[1], 0), 3.0)
        self.assertEqual(np.round(op.get('coef')[2], 0), 0.0)

    def test_predict_target_variable(self):
        """GLMNet: Test to predict a target variable."""
        df_ip = pd.read_csv(path + "test_glmnet.csv")
        mod = GLMNet(df=df_ip,
                     y_var=["y"],
                     x_var=["x1", "x2", "x3"])
        df_predict = pd.DataFrame({"x1": [10, 20],
                                   "x2": [5, 10],
                                   "x3": [100, 0]})
        op = mod.predict(df_predict)
        op = np.round(np.array(op["y"]), 1)
        exp_op = np.array([135.0, 170.0])
        self.assertEqual((op == exp_op).all(), True)


# =============================================================================
# --- Main
# =============================================================================

if __name__ == '__main__':
    unittest.main()
