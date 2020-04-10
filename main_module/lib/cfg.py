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

    Author: Diptesh Basak
    Date: Apr 10, 2020
    License: BSD 3-Clause
"""

# pylint: disable=invalid-name

import sys
import os
import socket
import re

__version__: str = "0.2.4"
__doc__: str = "Code library for python 3.7"
module: str = "main_module"

# Set environment
hdfs: bool = bool(re.match(r"[a-z0-9]+\.email\.com", socket.gethostname()))

# Set module path
path: str = os.path.abspath(os.path.dirname(sys.argv[0]))
path = re.sub(r"(.+)(\/" + module + ".*)", r"\1/", path)
