"""
Initialization file for unit tests.

Author
------
::

    Author: Diptesh Basak
    Date: May 15, 2019
    License: BSD 3-Clause
"""

# pylint: disable=invalid-name
# pylint: disable=wrong-import-position

import os
import sys
import re

# Set module path
path = os.path.abspath(os.path.dirname(sys.argv[0]))
path = re.sub(r"(.+)(\/test)", r"\1", path)

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))
import main_module  # noqa: F841

__all__ = ["main_module", ]
