"""
cfg.

Configuration file
------------------

Create module level variables for module ``lib``.

Input
-----

Change the following::

    __version__ : str
    __doc__     : str
    module      : str

Output
------

The file sets the following variables:

>>> __version__
>>> __doc__
>>> module
>>> hdfs
>>> path

Author
------
::

    Author: Diptesh
    Date: Mar 03, 2020
    License: BSD 3-Clause
"""

# pylint: disable=invalid-name

import sys
import os
import socket
import re

__version__ = "0.2.0"
__doc__ = "Code library for python 3.7"
module = "main_module"

# Set environment
hdfs = bool(re.match(r"[a-z0-9]+\.target\.com", socket.gethostname()))

# Set module path
path = os.path.abspath(os.path.dirname(sys.argv[0]))
path = re.sub(r"(.+)(\/" + module + ".*)", r"\1", path)
path = path + "/"
