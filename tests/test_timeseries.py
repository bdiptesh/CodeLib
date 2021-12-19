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
# pylint: disable=W0611

import unittest
import warnings
import re
import sys

from inspect import getsourcefile
from os.path import abspath

# Set base path
path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+)(\/tests.*)", "\\1", path)

sys.path.insert(0, path)

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


# =============================================================================
# --- Main
# =============================================================================

if __name__ == '__main__':
    unittest.main()
