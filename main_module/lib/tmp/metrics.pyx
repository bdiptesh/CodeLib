"""
Common metrics required in machine learning modules.

Available functions:
    - ``rsq``: R-Squared
    - ``rmse``: Root mean squared error
    - ``mse``: Mean squared error

Author
------
::

    Author: Diptesh Basak
    Date: Thu Apr 16, 2020
    License: BSD 3-Clause
"""

import numpy as _np

from cpython cimport array

# =============================================================================
# --- User defined functions
# =============================================================================

def rsq(list y, list y_hat):
    """
    Compute `Coefficient of determination
    <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_
    or R-Squared.

    Parameters
    ----------
    :y: list

        Actual values.

    :y_hat: list

        Predicted values.

    Returns
    -------
    :op: float

        R-Squared value.

    """
    return _np.round(_np.corrcoef(y, y_hat)[0][1] ** 2, 3)


def rmse(list y, list y_hat):
    """
    Compute `Root mean square error
    <https://en.wikipedia.org/wiki/Root-mean-square_deviation>`_.

    Parameters
    ----------
    :y: list

        Actual values.

    :y_hat: list

        Predicted values.

    Returns
    -------
    :op: float

        Root mean square error.

    """
    cdef array.array y_arr =  array.array("f", y)
    cdef array.array y_hat_arr =  array.array("f", y_hat)
    cdef long long int end_p = len(y)
    cdef array.array op =  array.array("f", [0.0] * end_p)
    cdef long long int i
    for i in range(0, end_p, 1):
        op[i] = (y_arr[i] - y_hat_arr[i]) ** 2
    return _np.round((sum(op) / end_p) ** 0.5, 3)


def mse(list y, list y_hat):
    """
    Compute `Mean squared error
    <https://en.wikipedia.org/wiki/Mean_squared_error>`_.

    Parameters
    ----------
    :y: list

        Actual values.

    :y_hat: list

        Predicted values.

    Returns
    -------
    :op: float

        Mean squared error.

    """
    cdef array.array y_arr =  array.array("f", y)
    cdef array.array y_hat_arr =  array.array("f", y_hat)
    cdef long long int end_p = len(y)
    cdef array.array op =  array.array("f", [0.0] * end_p)
    cdef long long int i
    for i in range(0, end_p, 1):
        op[i] = (y_arr[i] - y_hat_arr[i]) ** 2
    return _np.round(sum(op) / end_p, 3)
