"""
Test suite module for ``Cluster``.

Credits
-------
::

    Authors:
        - Diptesh

    Date: Sep 01, 2021
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

from mllib.lib.cluster import Cluster  # noqa: F841

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


class TestIntegrationCluster(unittest.TestCase):
    """Test suite for module ``metric``."""

    def setUp(self):
        """Set up for module ``metric``."""

    def test_categorical(self):
        y = [1, 2, 3]
        y_hat = [1, 5, 3]
        self.assertEqual(1, 1)


# =============================================================================
# --- Main
# =============================================================================

if __name__ == '__main__':
    unittest.main()
