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

from typing import List, Dict, Any

import re
import sys
import os

from inspect import getsourcefile
from os.path import abspath

import pandas as pd
import numpy as np

path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+\/)(.+.py)", "\\1", path)
sys.path.insert(0, path)

import metrics  # noqa: F841


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

    ds: str, optional

        Column name of the date variable (the default is None).

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
                         x_var=["cost", "stock_level", "retail_price"],
                         ds="ds")
    >>> df_op = mod.predict(x_predict)

    """

    def __init__(self,
                 df: pd.DataFrame,
                 y_var: str,
                 x_var: List[str] = None,
                 ds: str = "ds",
                 param: Dict = None):
        """Initialize variables."""
        self.y_var = y_var
        self.x_var = x_var
        self.ds = ds
        self.df = df.reset_index(drop=True)
        if param is None:
            param = {"interval_width": 0.95}
        self.model = None
        self.model_summary = None
        self.param = param
        self._pre_processing()
        self._fit()
        self._compute_metrics()

    def _pre_processing(self):
        pass

    def _opt_param(self):
        pass

    def _fit(self):
        pass

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
