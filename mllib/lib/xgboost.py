"""
Random Forest module.

**Available routines:**

- class ``XGBoost``: Builds XGBoost model using cross validation.

Credits
-------
::

    Authors:
        - Diptesh
        - Madhu

    Date: Sep 27, 2021
"""

# pylint: disable=invalid-name
# pylint: disable=R0902,R0903,R0913,C0413

from typing import List, Dict, Any

import re
import sys
from inspect import getsourcefile
from os.path import abspath

import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+\/)(.+.py)", "\\1", path)
sys.path.insert(0, path)

import metrics  # noqa: F841


class XGBoost():
    """Random forest module.

    Objective:
        - Build
          `XGBoost < https://en.wikipedia.org/wiki/XGBoost >'
          model and determine optimal k

    Parameters
    ----------
    df : pandas.DataFrame

        Pandas dataframe containing the `y_var` and `x_var`

    y_var : str

        Dependant variable

    x_var : List[str]

        Independant variables

    method : str, optional

        Can be either `classify` or `regression` (the default is regression)

    k_fold : int, optional

        Number of cross validations folds (the default is 5)

    param : dict, optional

        XGBoost parameters (the default is None).
        In case of None, the parameters will default to::

            n_estimators: [1000]
            learning_rate: [i/1000 for i in range(2, 11)]
            subsample: [i/10 for i in range(5, 10)]
            colsample_bytree: [i/10 for i in range(1, 11)]
            min_child_weight: list(range(1, 11))
            max_depth: [1, len(x_var)]
            gamma: list(range(1, len(x_var)))
            objective: ["reg:squarederror"]

    Returns
    -------
    model : object

        Final optimal model.

    best_params_ : Dict

        Best parameters amongst the given parameters.

    model_summary : Dict

        Model summary containing key metrics like R-squared, RMSE, MSE, MAE,
        MAPE for regression and Accuracy, Precision, Recall, F1 score for
        classification.

    Methods
    -------
    predict

    Example
    -------
    >>> mod = XGBoost(df=df_ip, y_var="y", x_var=["x1", "x2", "x3"])
    >>> df_op = mod.predict(df_predict)

    """

    def __init__(self,
                 df: pd.DataFrame,
                 y_var: str,
                 x_var: List[str],
                 method: str = "regression",
                 k_fold: int = 5,
                 param: Dict = None):
        """Initialize variables for module ``XGBoost``."""
        self.y_var = y_var
        self.x_var = x_var
        self.df = df.reset_index(drop=True)
        self.method = method
        self.model = None
        self.k_fold = k_fold
        self.seed = 1
        if param is None:
            param = {"n_estimators": [1000],
                           "learning_rate": [i/1000 for i in range(2, 11)],
                           "subsample": [i/10 for i in range(5, 10)],
                           "colsample_bytree": [i/10 for i in range(1, 11)],
                           "min_child_weight": list(range(1, 11)),
                           "max_depth": list(range(1, len(x_var))),
                           "gamma": list(range(0, 21)),
                           }
            if method == "classify":
                param["objective"] = ["binary:logistic"]
            elif method == "regression":
                param["objective"] = ["reg:squarederror"]
        self.param = param
        self.best_params_ = self._fit()
        self.model_summary = None
        self._compute_metrics()

    def _compute_metrics(self):
        """Compute commonly used metrics to evaluate the model."""
        y = self.df.loc[:, self.y_var].values.tolist()
        y_hat = list(self.model.predict(self.df[self.x_var]))
        if self.method == "regression":
            model_summary = {"rsq": np.round(metrics.rsq(y, y_hat), 3),
                             "mae": np.round(metrics.mae(y, y_hat), 3),
                             "mape": np.round(metrics.mape(y, y_hat), 3),
                             "rmse": np.round(metrics.rmse(y, y_hat), 3)}
            model_summary["mse"] = np.round(model_summary["rmse"] ** 2, 3)
        if self.method == "classify":
            class_report = classification_report(y,
                                                 y_hat,
                                                 output_dict=True,
                                                 zero_division=0)
            model_summary = class_report["weighted avg"]
            model_summary["accuracy"] = class_report["accuracy"]
            model_summary = {key: round(model_summary[key], 3)
                             for key in model_summary}
        self.model_summary = model_summary

    def _fit(self) -> Dict[str, Any]:
        """Fit XGBoost model."""
        if self.method == "classify":
            tmp_model = xgb.XGBClassifier(n_jobs=-1,
                                         verbosity=0,
                                         silent=True,
                                         random_state=self.seed,
                                         seed=self.seed)
        elif self.method == "regression":
            tmp_model = xgb.XGBRegressor(n_jobs=-1,
                                         verbosity=0,
                                         silent=True,
                                         random_state=self.seed,
                                         seed=self.seed)
        gs = GridSearchCV(estimator=tmp_model,
                          param_grid=self.param,
                          n_jobs=-1,
                          verbose=0,
                          refit=True,
                          return_train_score=True,
                          cv=self.k_fold)
        gs_op = gs.fit(self.df[self.x_var],
                       self.df[self.y_var])
        self.model = gs_op
        return gs_op.best_params_

    def predict(self, x_predict: pd.DataFrame) -> pd.DataFrame:
        """Predict values."""
        df_op = x_predict.copy(deep=True)
        y_hat = self.model.predict(x_predict)
        df_op.insert(loc=0, column=self.y_var, value=y_hat)
        return df_op
