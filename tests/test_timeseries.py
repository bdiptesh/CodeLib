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
# pylint: disable=W0511,W0611

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

from mllib.lib.timeseries import AutoArima  # noqa: F841

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


# TODO: Change integration tests.
class TestTimeSeries(unittest.TestCase):
    """Test suite for module ``TimeSeries``."""

    def setUp(self):
        """Set up for module ``TimeSeries``."""

    @ignore_warnings
    def test_multivariate(self):
        """TimeSeries: Test for multivariate."""
        df_ip = pd.read_csv(path + "test_time_series.csv")
        mod = AutoArima(df=df_ip,
                        y_var="y",
                        x_var=["cost", "stock_level", "retail_price"],
                        param={"max_p": 5,
                               "max_d": 2,
                               "max_q": 2,
                               "threshold": 0.05})
        op = mod.model_summary
        self.assertEqual(mod.opt_pdq, (1, 0, 1))
        self.assertEqual(1.0, op["rsq"])
        self.assertAlmostEqual(5.214, op["mae"], places=1)
        self.assertAlmostEqual(0.014, op["mape"], places=1)
        self.assertAlmostEqual(11.052, op["rmse"], places=1)
        self.assertAlmostEqual(122.147, op["mse"], places=1)

    @ignore_warnings
    def test_univariate(self):
        """TimeSeries: Test for univariate."""
        df_ip = pd.read_csv(path + "test_ts_passengers.csv")
        mod = AutoArima(df=df_ip,
                        y_var="Passengers",
                        param={"max_p": 5,
                               "max_d": 2,
                               "max_q": 2,
                               "threshold": 0.05})
        op = mod.predict()
        self.assertAlmostEqual(op["Passengers"].values[0], 471.038, places=1)


# =============================================================================
# --- Main
# =============================================================================

if __name__ == '__main__':
    unittest.main()
