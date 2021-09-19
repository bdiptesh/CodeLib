"""
Module for commonly used machine learning modelling algorithms.

**Available routines:**

- udf ``create_lag_vars``: Create lag variables for time series data.
- class ``GLMNet``: Builds GLMnet model using cross validation.

Credits
-------
::

    Authors:
        - Madhu
        - Diptesh

    Date: Sep 16, 2021
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
from sklearn.model_selection import TimeSeriesSplit as ts_split

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
                    lst_lag: List[int] = None,
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

    lst_lag : List[int]

        Lag values list (the default is None)

    n_interval : str, optional

        Column name of the time interval variable (the default is None).

    Returns
    -------
    pd.DataFrame

        Pandas dataframe containing `y_var`, lag variables (`lag_xx`) and
        `x_var`.

    """
    if n_interval is None:
        df = df.reset_index(drop=True)
    elif len(df) != (df[n_interval].max() - df[n_interval].min() + 1):
        sys.exit("Missing/duplicate time instance found in input data")
    else:
        df = df.sort_values(by=n_interval)
        df = df.reset_index(drop=True)
    y_lag = df[y_var].copy(deep=True)
    time_int = len(y_lag)
    if lst_lag is None:
        lst_lag = []
        while time_int > 8:
            time_int = int(np.floor(time_int/2))
            lst_lag.extend([time_int])
        lst_lag.extend([4, 3, 2, 1])
    for lag in lst_lag:
        y_lag.loc[:, "lag_" + str(lag)] = y_lag["y"].shift(lag)
    y_lag = y_lag.join(df[x_var])
    if n_interval:
        y_lag = y_lag.join(df[n_interval])
        y_lag = y_lag.set_index(n_interval)
    op = y_lag.dropna().reset_index(drop=True)
    return lst_lag, op


class GLMNet_ts():
    """GLMNet time series module.

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

    lst_lag : List[int]

        Lag values list (the default is None)

    n_interval : str, optional

        Column name of the time interval variable (the default is None).


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
                 lst_lag: List[int] = None,
                 n_interval: str = None,
                 param: Dict = None):
        """Initialize variables for module ``GLMNet``."""
        self.df = df[y_var + x_var]
        self.y_var = y_var
        self.x_var = x_var
        self.lst_lag = lst_lag
        self.n_interval = n_interval
        self.model_summary = None
        self.max_epoch = None
        if param is None:
            param = {"seed": 1,
                     "a_inc": 0.05,
                     "test_perc": 0.25,
                     "n_jobs": -1,
                     "k_fold": 10}
        self.param = param
        self.param["l1_range"] = list(np.round(np.arange(self.param["a_inc"],
                                                         1.01,
                                                         self.param["a_inc"]),
                                               2))
        self._fit()
        self._compute_metrics()

    def _fit(self) -> None:
        """Fit the best GLMNet time series model."""
        if self.n_interval is None:
            self.max_epoch = len(self.df) - 1
        else:
            self.max_epoch = self.df[self.n_interval].max()
        self.lag_var, df_ip = create_lag_vars(self.df,
                                              self.y_var,
                                              self.x_var,
                                              self.lst_lag,
                                              self.n_interval)
        self.x_var = list(df_ip.columns)
        self.x_var.remove(self.y_var[0])
        df_train = df_ip.iloc[0:int(len(df_ip) * (1-self.param["test_perc"]))]
        df_test = df_ip.iloc[int(len(df_ip) * (1-self.param["test_perc"])):]
        train_x = df_train[self.x_var]
        train_y = df_train[self.y_var]
        test_x = df_test[self.x_var]
        test_y = df_test[self.y_var]
        self.param["k_fold"] = ts_split(n_splits=self.param["k_fold"])
        self.param["k_fold"] = self.param["k_fold"].split(X=train_y)
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
        model_summary = {"rsq": np.round(metrics.rsq(y, y_hat), 3),
                         "mae": np.round(metrics.mae(y, y_hat), 3),
                         "mape": np.round(metrics.mape(y, y_hat), 3),
                         "rmse": np.round(metrics.rmse(y, y_hat), 3)}
        model_summary["mse"] = np.round(model_summary["rmse"] ** 2, 3)
        self.model_summary = model_summary

    def predict(self, df_predict: pd.DataFrame) -> pd.DataFrame:
        """Predict y_var/target variable.

        Parameters
        ----------
        df_predict : pd.DataFrame

            Pandas dataframe containing `x_var`, 'n_interval' (optional)

        Returns
        -------
        pd.DataFrame

            Pandas dataframe containing predicted `y_var` and `x_var`.

        """
        if self.n_interval is None:
            df_predict = df_predict.reset_index(drop=True)
            df_predict = \
                df_predict.set_index(df_predict.index+self.max_epoch+1)
        elif len(df_predict) != (df_predict[self.n_interval].max() \
                                 - df_predict[self.n_interval].min() + 1) \
                                or df_predict[self.n_interval].min() \
                                    > self.max_epoch+1:
            sys.exit("Missing time instance found in input data")
        else:
            df_ip = self.df[self.df[self.n_interval] \
                            <= df_predict[self.n_interval].min()]
            df_predict = df_predict.sort_values(by=self.n_interval)
            df_predict = df_predict.set_index(self.n_interval)
        df_predict = df_predict[self.x_var]
        df_predict["y"] = -1
        for i in range(0, len(df_predict)):
            # for i in range(0, len(df_ip)):
            df_pred = pd.DataFrame(df_predict.iloc[i])
            df_pred = df_pred.T # Transpose
            period_val = df_pred.index
            df_pred = df_pred[self.x_var].reset_index(drop=True)
            df_pred_x = pd.DataFrame(
                {"lag_"+str(self.lst_lag[0]): df_ip.iloc[len(df_ip)\
                                                         -self.lst_lag[0]]})
            for j in range(1, len(self.lst_lag)):
                df_tmp = pd.DataFrame(
                    {"lag_"+str(self.lst_lag[j]): \
                     df_ip.iloc[len(df_ip)-self.lst_lag[j]]})
                df_pred_x = df_pred_x.join(df_tmp)
            df_pred_x = df_pred_x.reset_index(drop=True)
            df_pred_x = df_pred_x.join(df_pred)
            y_hat = self.model.predict(df_pred_x)
            df_tmp = pd.DataFrame()
            df_tmp['y'] = y_hat
            df_ip = df_ip.append(df_tmp).reset_index(drop=True)
            df_predict.loc[period_val, "y"] = y_hat
        return df_predict
