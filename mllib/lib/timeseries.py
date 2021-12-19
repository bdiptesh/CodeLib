"""
Time series module.

**Available routines:**

- class ``TimeSeries``: Builds time series model using fbprophet.

Credits
-------
::

    Authors:
        - Diptesh
        - Madhu

    Date: Dec 18, 2021
"""

# pylint: disable=invalid-name
# pylint: disable=wrong-import-position
# pylint: disable=R0903

from typing import List, Dict

import re
import sys

from inspect import getsourcefile
from os.path import abspath

import pandas as pd

path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+\/)(.+.py)", "\\1", path)
sys.path.insert(0, path)


class TimeSeries():
    """Time series module.

    Parameters
    ----------
    df: pandas.DataFrame

        Pandas dataframe containing the `y_var`, `ds` and `x_var`

    y_var: str

        Dependant variable

    x_var: List[str], optional

        Independant variables (the default is None).

    param: dict, optional, Not implemented yet

        Time series parameters (the default is None).

    Returns
    -------
    model: object

        Final optimal model.

    model_summary: Dict

        Model summary containing key metrics like R-squared, RMSE, MSE, MAE,
        MAPE.

    Methods
    -------
    predict

    Example
    -------
    >>> mod = TimeSeries(df=df_ip,
                         y_var="y",
                         x_var=["cost", "stock_level", "retail_price"])
    >>> df_op = mod.predict(x_predict)

    """

    def __init__(self,
                 df: pd.DataFrame,
                 y_var: str,
                 x_var: List[str] = None,
                 param: Dict = None):
        """Initialize variables."""
        self.df = df
        self.y_var = y_var
        self.x_var = x_var
        self._check_data()
        self.param = param

    def _check_data(self):
        df_check = self.df.dropna()
        if len(self.df) != len(df_check):
            raise ValueError("Found missing values in input data")
        # TO DO: Check for y_var, x_var, number of observations.

    def _opt_param(self):
        """Determine optimal parameters."""

    def _fit(self):
        """Fit the model."""

    def predict(self,
                x_predict: pd.DataFrame = None,
                n_interval: int = 1) -> pd.DataFrame:
        """Predict module.

        Parameters
        ----------
        x_predict : pd.DataFrame, optional

            Pandas dataframe containing `ds` and `x_var` (the default is None).

        n_interval : int, optional

            Number of time period to predict (the default is 1).

        Returns
        -------
        pd.DataFrame

            Pandas dataframe containing `y_var`, `ds` and `x_var`.

        """
