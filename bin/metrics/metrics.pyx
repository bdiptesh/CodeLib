"""
Common metrics required in machine learning modules.

**Available functions:**
    - ``rsq``: R-Squared
    - ``mse``: Mean squared error
    - ``rmse``: Root mean squared error
    - ``mae``: Mean absolute error
    - ``mape``: Mean absolute percentage error

Credits
-------
::

    Authors:
        - Diptesh

    Date: Sep 10, 2021
"""

import numpy as _np

# =============================================================================
# --- User defined functions
# =============================================================================


cpdef rsq(list y, list y_hat):
    """
    Compute `Coefficient of determination
    <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_
    or R-Squared.

    Parameters
    ----------
    y : list

        Actual values.

    y_hat : list

        Predicted values.

    Returns
    -------
    op : float

        R-Squared value.

    """
    return _np.round(_np.corrcoef(y, y_hat)[0][1] ** 2, 3)


cpdef mse(list y, list y_hat):
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
    cdef int i
    cdef int arr_len
    cdef double a
    cdef double b
    cdef double op = 0.0
    arr_len = len(y)
    for i in range(0, arr_len, 1):
        a = y[i]
        b = y_hat[i]
        op = op + (a - b) ** 2
    op = op * arr_len ** -1.0
    return op

cpdef rmse(list y, list y_hat):
    """
    Compute `Root mean square error
    <https://en.wikipedia.org/wiki/Root-mean-square_deviation>`_.

    Parameters
    ----------
    y : list

        Actual values.

    y_hat : list

        Predicted values.

    Returns
    -------
    op : float

        Root mean square error.

    """
    return mse(y, y_hat) ** 0.5


cpdef mae(list y, list y_hat):
    """
    Compute `Mean absolute error
    <https://en.wikipedia.org/wiki/Mean_absolute_error>`_.

    Parameters
    ----------
    y : list

        Actual values.

    y_hat : list

        Predicted values.

    Returns
    -------
    op : float

        Mean absolute error.

    """
    cdef int i
    cdef int arr_len
    cdef double a
    cdef double b
    cdef double op = 0.0
    arr_len = len(y)
    for i in range(0, arr_len, 1):
        a = y[i]
        b = y_hat[i]
        op += abs(a - b)
    op = op * arr_len ** -1.0
    return op


cpdef mape(list y, list y_hat):
    """
    Compute `Mean absolute percentage error
    <https://en.wikipedia.org/wiki/Mean_absolute_percentage_error>`_.

    Parameters
    ----------
    y : list

        Actual values.

    y_hat : list

        Predicted values.

    Returns
    -------
    op : float

        Mean absolute percentage error.

    """
    cdef int i
    cdef int arr_len
    cdef double a
    cdef double b
    cdef double op = 0.0
    arr_len = len(y)
    for i in range(0, arr_len, 1):
        a = y[i]
        b = y_hat[i]
        op += abs(1 - (b * a ** -1.0))
    op = op * arr_len ** -1.0
    return op