"""
Test suite module for ``timeseries``.

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

from mllib.lib.timeseries import TimeSeries  # noqa: F841

# =============================================================================
# --- DO NOT CHANGE ANYTHING FROM HERE
# =============================================================================

path = path + "/data/input/"

# =============================================================================
# --- User defined functions
# =============================================================================


def ignore_warnings(test_func):
    """Suppress warnings."""

    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)
    return do_test


class TestTimeSeries(unittest.TestCase):
    """Test suite for module ``TimeSeries``."""

    def setUp(self):
        """Set up for module ``TimeSeries``."""

    @ignore_warnings
    def test_multivariate(self):
        """TimeSeries: Test for multivariate."""
        df_ip = pd.read_excel(path + "test_time_series.xlsx",
                              sheet_name="product_01")
        mod = TimeSeries(df=df_ip,
                         y_var="y",
                         x_var=["cost", "stock_level", "retail_price"],
                         ds="ds")
        op = mod.model_summary
        self.assertAlmostEqual(0.99, op["rsq"], places=1)

    @ignore_warnings
    def test_raise_exceptions(self):
        """TimeSeries: Test raise exceptions."""
        df_ip = pd.read_excel(path + "test_time_series.xlsx",
                              sheet_name="product_01")
        self.assertRaises(NotImplementedError, TimeSeries,
                          df=df_ip,
                          y_var="y",
                          x_var=["stock_level", "retail_price"],
                          ds="ds",
                          uid="cost")
        self.assertRaises(NotImplementedError, TimeSeries,
                          df=df_ip,
                          y_var="y",
                          x_var=["stock_level", "retail_price"],
                          ds="ds",
                          k_fold=5)

    def test_univariate(self):
        """TimeSeries: Test for univariate."""
        df_ip = pd.read_csv(path + "test_ts_passengers.csv")
        mod = TimeSeries(df=df_ip, y_var="Passengers", ds="Month")
        op = mod.predict()
        self.assertAlmostEqual(op["y"].values[0], 446.911, places=1)


# =============================================================================
# --- Main
# =============================================================================

if __name__ == '__main__':
    unittest.main()
