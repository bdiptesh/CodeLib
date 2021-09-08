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
# pylint: disable=R0902,R0903,R0913,R0914

from typing import List, Dict

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit as ts_split
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

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

    timeseries : bool, optional

        Boolean value to indicate time-series inputs (the default is False).

    search_method : str, optional

        String to indicate the hyper parameter search method. Possible values
        are "grid", "random" (the default is "random")

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
            lambda_param: [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
            timeseries: False
            search_method: "random"

    """

    def __init__(self,
                 df: pd.DataFrame,
                 y_var: List[str],
                 x_var: List[str],
                 timeseries: bool = False,
                 search_method: str = "random",
                 n_interval: str = None,
                 param: Dict = None):
        """Initialize variables for module ``GLMNet``."""
        self.df = df[y_var].join(df[x_var])
        self.y_var = y_var
        self.x_var = x_var
        self.model = None
        self.n_interval = n_interval
        if param is None:
            param = {"seed": 1,
                     "a_inc": 0.05,
                     "test_perc": 0.25,
                     "n_jobs": -1,
                     "k_fold": 10,
                     "lambda_param": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
                                      1.0, 10.0, 100.0]}
        self.param = param
        self.param["l1_range"] = list(np.round(np.arange(0.0, 1.01,
                                                         self.param["a_inc"]),
                                               10))
        self.param["timeseries"] = timeseries
        self.param["search_method"] = search_method

    def fit(self):
        """Fit the best GLMNet model."""
        train_x = self.df[self.x_var]
        train_x = pd.get_dummies(data=train_x, drop_first=True)
        train_y = self.df[self.y_var]
        if self.param["timeseries"]:
            folds = ts_split(n_splits=self.param["k_fold"])
            folds = folds.split(X=train_y)
        else:
            folds = self.param["k_fold"]
        est_glmnet = ElasticNet(random_state=self.param["seed"])
        grid = {"l1_ratio": self.param["l1_range"],
                "alpha": self.param["lambda_param"]}
        if self.param["search_method"] == "grid":
            self.model = GridSearchCV(estimator=est_glmnet,
                                      param_grid=grid,
                                      n_jobs=-1,
                                      cv=folds,
                                      verbose=0,
                                      scoring="neg_mean_squared_error")
            self.model.fit(train_x, train_y)
        if self.param["search_method"] == "random":
            # n_iter =  30% of grid hyper parameter combinations
            sample_perc = 0.3
            n_iter = int(np.ceil(len(self.param["l1_range"])
                                 * len(self.param["lambda_param"])
                                 * sample_perc))
            self.model = RandomizedSearchCV(estimator=est_glmnet,
                                            param_distributions=grid,
                                            n_jobs=self.param["n_jobs"],
                                            n_iter=n_iter,
                                            cv=folds,
                                            verbose=1,
                                            scoring="neg_mean_squared_error")
            self.model.fit(train_x, train_y)

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
        df_predict_cp = df_predict.copy(deep=True)
        df_predict = pd.get_dummies(data=df_predict, drop_first=True)
        df_op = pd.DataFrame(self.model.predict(df_predict))
        df_op.columns = ["y_hat"]
        df_op = df_op.join(df_predict_cp)
        return df_op
