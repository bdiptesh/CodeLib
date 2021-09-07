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

# Set base path
path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+)(\/tests.*)", "\\1", path)

sys.path.insert(0, path)

from mllib.lib.model import create_lag_vars  # noqa: F841

# =============================================================================
# --- DO NOT CHANGE ANYTHING FROM HERE
# =============================================================================

path = path + "/data/input/"

# =============================================================================
# --- User defined functions
# =============================================================================


def ignore_warnings(test_func):
    """Suppress deprecation warnings of pulp."""

    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)
    return do_test


class TestCreateLagVars(unittest.TestCase):
    """Test suite for module ``metric``."""

    def setUp(self):
        """Set up for module ``metric``."""

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


# =============================================================================
# --- Main
# =============================================================================


if __name__ == '__main__':
    unittest.main()
