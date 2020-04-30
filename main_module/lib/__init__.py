"""
Initialization file for main_module lib.

Author
------
::

    Author: Diptesh Basak
    Date: Apr 18, 2020
    License: BSD 3-Clause
"""

# pylint: disable=invalid-name

import os
import sys

# Set module path
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))
