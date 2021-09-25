"""
Test suite module for ``glmnet_ts``.

Credits
-------
::

    Authors:
        - Madhu
        - Diptesh

    Date: Sep 24, 2021
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
import pytest

# Set base path
path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+)(\/tests.*)", "\\1", path)

sys.path.insert(0, path)

from mllib.lib.glmnet_ts import create_lag_vars  # noqa: F841
from mllib.lib.glmnet_ts import GLMNet_ts  # noqa: F841

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
        lst_lag, df_op = create_lag_vars(df=df_ip,
                                         y_var=["y"],
                                         x_var=["x1", "x2"])
        exp_op = df_ip[list(df_ip.columns[1:])].dropna().reset_index(drop=True)
        self.assertEqual(df_op.equals(exp_op), True)
        self.assertEqual([6, 4, 3, 2, 1], lst_lag)

    def test_interval_specified(self):
        """Lag vars: Test when interval is specified."""
        df_ip = pd.read_csv(path + "test_lag_var.csv")
        lst_lag, df_op = create_lag_vars(df=df_ip,
                                         y_var=["y"],
                                         x_var=["x1", "x2"],
                                         n_interval="week")
        exp_op = df_ip[list(df_ip.columns[1:])].dropna().reset_index(drop=True)
        self.assertEqual(df_op.equals(exp_op), True)
        self.assertEqual([6, 4, 3, 2, 1], lst_lag)

    def test_lag_vars_specified(self):
        """Lag vars: Test when lags are specified."""
        df_ip = pd.read_csv(path + "test_lag_var.csv")
        lst_lag, df_op = create_lag_vars(df=df_ip,
                                         y_var=["y"],
                                         x_var=["x1", "x2"],
                                         lst_lag=[3, 2, 1])
        exp_op = df_ip.iloc[:, [1, 4, 5, 6, 7, 8]]\
            .dropna().reset_index(drop=True)
        self.assertEqual(df_op.equals(exp_op), True)
        self.assertEqual([3, 2, 1], lst_lag)


class TestGLMNet_ts(unittest.TestCase):
    """Test suite for module ``GLMNet_ts``."""

    def setUp(self):
        """Set up for module ``GLMNet_ts``."""

    def test_known_equation(self):
        """GLMNet_ts: Test a known equation with/without n_interval."""
        df_ip = pd.read_csv(path + "test_glmnet_ts1.csv")
        df_train_ip = df_ip.iloc[0:len(df_ip)]
        mod = GLMNet_ts(df=df_train_ip,
                        y_var=["y"],
                        x_var=["x1", "x2"],
                        lst_lag=[3, 1])
        op = mod.opt
        self.assertTrue(0.5 <= np.round(op.get('intercept'), 0) <= 1.5)
        self.assertTrue(0.15 <= np.round(op.get('coef')[0], 2) <= 0.25)
        self.assertTrue(0.65 <= np.round(op.get('coef')[1], 2) <= 0.75)
        self.assertTrue(0.75 <= np.round(op.get('coef')[2], 2) <= 0.85)
        self.assertTrue(0.45 <= np.round(op.get('coef')[3], 2) <= 0.55)
        mod = GLMNet_ts(df=df_train_ip,
                        y_var=["y"],
                        x_var=["x1", "x2"],
                        lst_lag=[3, 1],
                        n_interval="week")
        op = mod.opt
        self.assertTrue(0.5 <= np.round(op.get('intercept'), 0) <= 1.5)
        self.assertTrue(0.15 <= np.round(op.get('coef')[0], 2) <= 0.25)
        self.assertTrue(0.65 <= np.round(op.get('coef')[1], 2) <= 0.75)
        self.assertTrue(0.75 <= np.round(op.get('coef')[2], 2) <= 0.85)
        self.assertTrue(0.45 <= np.round(op.get('coef')[3], 2) <= 0.55)

    def test_predict_target_variable(self):
        """GLMNet_ts: Test predictor with/without n_interval."""
        df_ip = pd.read_csv(path + "test_glmnet_ts1.csv")
        # without n_interval
        df_train_ip = df_ip.iloc[0:95]
        mod = GLMNet_ts(df=df_train_ip,
                        y_var=["y"],
                        x_var=["x1", "x2"],
                        lst_lag=[3, 1])
        op = mod.opt
        df_predict = df_ip.iloc[95:len(df_ip)]
        y_pred = mod.predict(df_predict)
        y_pred = np.round(np.array(y_pred["y"]), 1)
        df_exp = df_ip.copy(deep=True)
        df_exp['lag_3'] = df_exp["y"].shift(3)
        df_exp['lag_1'] = df_exp["y"].shift(1)
        df_exp = df_exp[["lag_3", "lag_1", "x1", "x2"]]
        df_exp = df_exp.iloc[95:len(df_ip)]
        df_exp["y"] = op.get('intercept')\
            + op.get('coef')[0] * df_exp["lag_3"]\
            + op.get('coef')[1] * df_exp["lag_1"]\
            + op.get('coef')[2] * df_exp["x1"]\
            + op.get('coef')[3] * df_exp["x2"]
        y_exp = np.round(np.array(df_exp["y"]), 1)
        for i, j in zip(y_pred, y_exp):
            self.assertTrue(j - 0.1 <= i <= j + 0.1)
        # with n_interval
        mod = GLMNet_ts(df=df_train_ip,
                        y_var=["y"],
                        x_var=["x1", "x2"],
                        lst_lag=[3, 1],
                        n_interval="week")
        op = mod.opt
        df_predict = df_ip.iloc[95:len(df_ip)]
        y_pred = mod.predict(df_predict)
        y_pred = np.round(np.array(y_pred["y"]), 1)
        df_exp = df_ip.copy(deep=True)
        df_exp['lag_3'] = df_exp["y"].shift(3)
        df_exp['lag_1'] = df_exp["y"].shift(1)
        df_exp = df_exp[["lag_3", "lag_1", "x1", "x2"]]
        df_exp = df_exp.iloc[95:len(df_ip)]
        df_exp["y"] = op.get('intercept')\
            + op.get('coef')[0] * df_exp["lag_3"]\
            + op.get('coef')[1] * df_exp["lag_1"]\
            + op.get('coef')[2] * df_exp["x1"]\
            + op.get('coef')[3] * df_exp["x2"]
        y_exp = np.round(np.array(df_exp["y"]), 1)
        for i, j in zip(y_pred, y_exp):
            self.assertTrue(j - 0.1 <= i <= j + 0.1)

    @staticmethod
    def test_for_exit():
        """GLMNet_ts: Test for missing time instance."""
        df_ip = pd.read_csv(path + "test_glmnet_ts1.csv")
        # without n_interval
        df_train_ip = df_ip.iloc[0:95]
        mod = GLMNet_ts(df=df_train_ip,
                        y_var=["y"],
                        x_var=["x1", "x2"],
                        lst_lag=[3, 1],
                        n_interval="week")
        df_predict = df_ip.iloc[96:len(df_ip)]
        with pytest.raises(SystemExit):
            df_predict = mod.predict(df_predict)


# =============================================================================
# --- Main
# =============================================================================

if __name__ == '__main__':
    unittest.main()
