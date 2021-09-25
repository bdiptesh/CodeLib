"""
Module for commonly used machine learning modelling algorithms.

**Available routines:**

- class ``Knn``: Builds K-Nearest Neighnour model sing cross validation.

Credits
-------
::

    Authors:
        - Diptesh
        - Madhu

    Date: Sep 25, 2021
"""

# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-few-public-methods

from typing import List, Dict, Any

import re
import sys
from inspect import getsourcefile
from os.path import abspath

import pandas as pd

from sklearn import neighbors as sn
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV

path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+\/)(.+.py)", "\\1", path)
sys.path.insert(0, path)

class Knn():
    """ K-Nearest Neighbour (KNN) module.

    Objective:
    - Build KNN model and determine optimal k

    Parameters
    ----------
    :df: pandas.DataFrame

        Pandas dataframe containing the `y_var` and `x_var`

    :y_var: str

        Target variable

    :x_var: list

        List containing independant variables

    :method: str, optional

        Can be either `classify` or `regression` (default is 'classify')

    :k_fold: int, optional

        Number of cross validations folds (default is 5)

    :param: dict, optional

        KNN parameters (the default is None).
        In case of None, the parameters will default to::

            n_neighbors: max(int(len(df)/(k_fold * 2)), 1)
            weights: ["uniform", "distance"]
            metric: ["euclidean", "manhattan"]

    Methods
    -------
    predict

    Example
    -------
    >>> mod = Knn(df=df_ip, y_var=["y"], x_var=["x1", "x2", "x3"])
    >>> df_op = mod.predict(df_predict)

    """

    def __init__(self,
                 df: pd.DataFrame,
                 y_var: str,
                 x_var: List[str],
                 method: str = "classify",
                 k_fold: int = 5,
                 param: Dict = None):
        """Initialize variables for module ``Knn``."""
        self.df = df.reset_index(drop=True)
        self.y_var = y_var
        self.x_var = x_var
        self.method = method
        self.model = None
        self.k_fold = k_fold
        if param is None:
            max_k = max(int(len(self.df)/(self.k_fold * 2)), 1)
            param = {"n_neighbors": list(range(1, max_k, 2)),
                     "weights": ["uniform", "distance"],
                     "metric": ["euclidean", "manhattan"]}
        self.param = param
        self._pre_process()
        self._fit()

    def _pre_process(self):
        """Pre-process the data, one hot encoding and scaling."""
        df_ip_x = pd.get_dummies(self.df[self.x_var])
        self.x_var = list(df_ip_x.columns)
        df_ip_x = pd.DataFrame(scale(df_ip_x))
        df_ip_x.columns = self.x_var
        self.df = self.df[[self.y_var]].join(df_ip_x)

    def _fit(self) -> Dict[str, Any]:
        """Fit KNN model."""
        if self.method == "classify":
            gs = GridSearchCV(sn.KNeighborsClassifier(),
                              self.param,
                              verbose=0,
                              cv=self.k_fold,
                              n_jobs=1)
        elif self.method == "regression":
            gs = GridSearchCV(sn.KNeighborsRegressor(),
                              self.param,
                              verbose=0,
                              cv=self.k_fold,
                              n_jobs=1)
        gs_op = gs.fit(self.df[self.x_var],
                       self.df[self.y_var])
        opt_k = gs_op.best_params_.get("n_neighbors")
        weight = gs_op.best_params_.get("weights")
        metric = gs_op.best_params_.get("metric")
        if self.method == "classify":
            model = sn.KNeighborsClassifier(n_neighbors=opt_k,
                                            weights=weight,
                                            metric=metric)
        elif self.method == "regression":
            model = sn.KNeighborsRegressor(n_neighbors=opt_k,
                                           weights=weight,
                                           metric=metric)
        self.model = model.fit(self.df[self.x_var],
                               self.df[self.y_var])
        return gs_op.best_params_

    def predict(self, x_pred: pd.DataFrame) -> pd.DataFrame:
        """Prediction module."""
        x_pred = pd.DataFrame(scale(pd.get_dummies(x_pred)))
        return self.model.predict(x_pred)
