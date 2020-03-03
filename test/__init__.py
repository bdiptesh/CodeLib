"""
Initialization file for unit tests.

Author
------
::

    Author: Diptesh.Basak
    Date: May 15, 2019
    License: BSD 3-Clause
"""

# pylint: disable=invalid-name
# pylint:disable=wrong-import-position

import os
import sys
import re

# Set module path
path = os.path.abspath(os.path.dirname(sys.argv[0]))
path = re.sub(r"(.+)(\/test)", r"\1", path)

sys.path.insert(0, path)

# Import modules

from main_module.lib.stat import Model
from main_module.lib.stat import Cluster
from main_module.lib.stat import Knn
from main_module.lib.stat import RandomForest
from main_module.lib.opt import TSP
from main_module.lib.opt import Transport
