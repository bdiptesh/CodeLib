"""
Module for commonly used machine learning modelling algorithms.

**Available routines:**

- udf ``create_lag_vars``: Create lag variables for time series data.
- class ``GLMNet``: Builds GLMnet model using cross validation.

Credits
-------
::

    Authors:
        - Diptesh

    Date: Sep 06, 2021
"""

# pylint: disable=invalid-name
# pylint: disable=R0902,R0903,R0913,R0914,C0413

from typing import List, Dict

import re
import sys
from inspect import getsourcefile
from os.path import abspath

import pandas as pd
import numpy as np

from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split as split

path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+\/)(.+.py)", "\\1", path)
sys.path.insert(0, path)

import metrics  # noqa: F841

# =============================================================================
# --- DO NOT CHANGE ANYTHING FROM HERE
# =============================================================================


def create_lag_vars(df: pd.DataFrame,
                    y_var: List[str],
                    x_var: List[str],
                    n_interval: str = None) -> pd.DataFrame:
    """Create lag variables for time series data.

    Parameters
    ----------
    df : pd.DataFrame

        Pandas dataframe containing `y_var`, `x_var` and `n_interval`
        (if provided).

    y_var : List[str]

        Dependant variable.

    x_var : List[str]

        Independant variables.

    n_interval : str, optional

        Column name of the time interval variable (the default is None).

    Returns
    -------
    pd.DataFrame

        Pandas dataframe containing `y_var`, lag variables (`lag_xx`) and
        `x_var`.

    """
    if n_interval is None:
        y_lag = df[y_var].reset_index(drop=True)
    else:
        y_lag = df.sort_values(by=n_interval)
        y_lag = y_lag[y_var].reset_index(drop=True)
    time_int = len(y_lag)
    lag_interval = []
    while time_int > 8:
        time_int = int(np.floor(time_int/2))
        lag_interval.extend([time_int])
    lag_interval.extend([4, 3, 2, 1])
    for lag in lag_interval:
        y_lag.loc[:, "lag_" + str(lag)] = y_lag["y"].shift(lag)
    y_lag = y_lag.join(df[x_var])
    op = y_lag.dropna().reset_index(drop=True)
    return op


class GLMNet():
    """GLMNet module.

    Objective:
        - Build
          `GLMNet <https://web.stanford.edu/~hastie/Papers/glmnet.pdf>`_
          model using optimal alpha and lambda

    Parameters
    ----------
    df : pd.DataFrame

        Pandas dataframe containing `y_var` and `x_var` variables.

    y_var : List[str]

        Dependant variable.

    x_var : List[str]

        Independant variables.

    strata : pd.DataFrame, optional

        A pandas dataframe column defining the strata (the default is None).

    param : Dict, optional

        GLMNet parameters (the default is None).
        In case of None, the parameters will default to::

            seed: 1
            a_inc: 0.05
            test_perc: 0.25
            n_jobs: -1
            k_fold: 10

    """

    def __init__(self,
                 df: pd.DataFrame,
                 y_var: List[str],
                 x_var: List[str],
                 strata: str = None,
                 param: Dict = None):
        """Initialize variables for module ``GLMNet``."""
        self.df = df[y_var + x_var]
        self.y_var = y_var
        self.x_var = x_var
        self.strata = strata
        self.model_summary = None
        if param is None:
            param = {"seed": 1,
                     "a_inc": 0.05,
                     "test_perc": 0.25,
                     "n_jobs": -1,
                     "k_fold": 10}
        self.param = param
        self.param["l1_range"] = list(np.round(np.arange(0.0001, 1.01,
                                                         self.param["a_inc"]),
                                               2))
        self._fit()
        self._compute_metrics()

    def _fit(self) -> None:
        """Fit the best GLMNet model."""
        train_x, test_x,\
            train_y, test_y = split(self.df[self.x_var],
                                    self.df[self.y_var],
                                    test_size=self.param["test_perc"],
                                    random_state=self.param["seed"],
                                    stratify=self.strata)
        mod = ElasticNetCV(l1_ratio=self.param["l1_range"],
                           fit_intercept=True,
                           alphas=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
                                   1.0, 10.0, 100.0],
                           normalize=True,
                           cv=self.param["k_fold"],
                           n_jobs=self.param["n_jobs"],
                           random_state=self.param["seed"])
        mod.fit(train_x, train_y.values.ravel())
        opt = {"alpha": mod.l1_ratio_,
               "lambda": mod.alpha_,
               "intercept": mod.intercept_,
               "coef": mod.coef_,
               "train_v": mod.score(train_x, train_y),
               "test_v": mod.score(test_x, test_y)}
        self.model = mod
        self.opt = opt

    def _compute_metrics(self):
        """Compute commonly used metrics to evaluate the model."""
        y = self.df[self.y_var].iloc[:, 0].values.tolist()
        y_hat = list(self.predict(self.df[self.x_var])["y"].values)
        model_summary = {"rsq": metrics.rsq(y, y_hat),
                         "mae": metrics.mae(y, y_hat),
                         "mape": metrics.mape(y, y_hat),
                         "rmse": metrics.rmse(y, y_hat)}
        model_summary["mse"] = model_summary["rmse"] ** 2
        self.model_summary = model_summary

    def predict(self, df_predict: pd.DataFrame) -> pd.DataFrame:
        """Predict y_var/target variable.

        Parameters
        ----------
        df_predict : pd.DataFrame

            Pandas dataframe containing `x_var`.

        Returns
        -------
        pd.DataFrame

            Pandas dataframe containing predicted `y_var` and `x_var`.

        """
        y_hat = self.model.predict(df_predict)
        df_predict = df_predict.copy()
        df_predict["y"] = y_hat
        return df_predict
