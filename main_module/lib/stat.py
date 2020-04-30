"""
Module for commonly used machine learning algorithms.

Statistics module
-----------------

**Available routines:**

- class ``Model``: Builds GLMnet model
- class ``Cluster``: K-means clustering
- class ``Knn``: K Nearest Neighbor
- class ``RandomForest``: Random forest
- class ``XGBoost``: XGBoost (random or timeseries split cross validation)

Objective
---------

- GLMnet model with cross validation
- Determining optimal number of clusters for given data
- Determine optimal number of `k` for classification/regression prediction
- Fit best random forest model for classification/regression prediction
- Fit best XGBoost model for regression prediction

Author
------
::

    Author: Diptesh Basak
    Date: Apr 17, 2020
    License: BSD 3-Clause
"""

# =============================================================================
# --- DO NOT CHANGE ANYTHING FROM HERE
# =============================================================================

# pylint: disable=invalid-name
# pylint: disable-msg=too-many-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=wrong-import-position
# pylint: disable=import-error

# =============================================================================
# --- Import libraries
# =============================================================================

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union, Any

import os
import sys
import copy
import pandas as pd
import numpy as np
import xgboost as xgb

import sklearn.ensemble as rf

from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split as split
from sklearn.model_selection import TimeSeriesSplit as tssplit
from sklearn import neighbors as sn
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.model_selection import RandomizedSearchCV

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))

from lib import metrics  # noqa: E402

# =============================================================================
# --- User defined functions
# =============================================================================


@dataclass
class Model():
    """
    Fit best model using k-fold GLMnet.

    Parameters
    ----------
    :seed: int, `optional`, `default:` ``123456789``

        Random seed

    :a_inc: float, `optional`, `default:` ``0.05``

        Interval for alpha values

    :test_perc: float, `optional`, `default:` ``0.25``

        Percent of data to be used for test

    :n_jobs: int, `optional`, `default:` ``-1``

        Number of cores to be used for multiprocessing

    :k_fold: int, `optional`, `default:` ``10``

        Number of folds for cross validation

    """

    seed: int = 123456789
    a_inc: float = 0.05
    test_perc: float = 0.25
    n_jobs: int = -1
    k_fold: int = 10

    def cv_glmnet(self,
                  df: pd.DataFrame,
                  y_var: List[str],
                  x_var: List[str],
                  strata: str = None
                  ) -> Dict[str, float]:
        """
        Glmnet cross validation module (sklearn).

        Parameters
        ----------
        :df: pandas.DataFrame

            Pandaas data frame containing **y_var** and **x_var**

        :y_var: list

            List containing the dependant variable

        :x_var: list

            List containing the independant variable

        :strata: pandas.DataFrame column name, `optional`, `default:` ``None``

            A pandas dataframe column defining the strata

        Returns
        -------
        :opt: dict

            Containing::

                alpha: Optimal alpha
                intercept: Intercept value
                coef: Coefficients
                train_v: R-Squared of training data
                test_v: R-Squared of test data

        :mod: object

            ElasticNetCV object

        """
        train_x, test_x, train_y, test_y = split(df[x_var],
                                                 df[y_var],
                                                 test_size=self.test_perc,
                                                 random_state=self.seed,
                                                 stratify=strata)
        inc = self.a_inc * 100
        l1_range = list(range(5, 105, int(inc)))
        l1_range[:] = [x / 100.0 for x in l1_range]
        mod = ElasticNetCV(l1_ratio=l1_range, fit_intercept=True,
                           normalize=True, cv=self.k_fold,
                           n_jobs=self.n_jobs, random_state=self.seed)
        mod.fit(train_x, train_y)
        opt = {'alpha': mod.l1_ratio_,
               'intercept': mod.intercept_,
               'coef': mod.coef_,
               'train_v': mod.score(train_x, train_y),
               'test_v': mod.score(test_x, test_y)}
        return opt, mod


@dataclass
class Cluster():
    """
    Clustering module.

    To determine optimal **k** using KMeans clustering and gap statistic.

    Parameters
    ----------
    :x_var: list

        List of clustering variables. All variables in **xVars** should be
        present in pandas.DataFrame **df**

    :max_clus: int

        Maximum number of clusters allowed. If **max_clus** is greater than
        the count of unique observations in pandas.DataFrame **df**, then
        minimum of **max_clus** and length of unique values in df is
        considered.

    Returns
    -------
    :df_op: pandas.DataFrame

        A pandas dataframe containing the input pandas dataframe **df** and
        cluster labels.

    """

    def __init__(self,
                 x_var: List[str],
                 max_clus: int):
        """Initialize variables for module ``Cluster``."""
        self.x_var = x_var
        self.max_clus = max_clus
        self.gap_val = None
        self.df_ip = None
        self.clus_ip = None
        self.df_op = None

    @staticmethod
    def _Compact(data: np.array,
                 centers: List[float],
                 labels: List[int]
                 ) -> float:
        """Compute compactness of given clustering solution."""
        k_sum = 0.0
        for i in enumerate(data):
            cluster_num = labels.item(i[0])
            k_sum += np.linalg.norm(centers[cluster_num] - data[i[0]])**2
        return k_sum

    def _NRef(self) -> pd.DataFrame:
        """
        Create null reference distribution.

        Parameters
        ----------
        :*self*.df_ip: pandas.DataFrame

            An input pandas dataframe for which null reference distribution
            needs to be created

        :*self*.clusVars: list

            A list of column names

        Returns
        -------
        :df_sample: pandas.DataFrame

            A pandas dataframe containing null reference distribution for given
            *self*.clusVars

        """
        df = self.df_ip[self.x_var]
        x_cat = df.select_dtypes(include=['object', 'bool'])
        x_num = df.select_dtypes(include=['int', 'float64'])
        if not x_cat.empty:
            for i, c in enumerate(x_cat.columns):
                cat_val_list = df[c].unique()
                uniqu_val = len(cat_val_list)
                temp_cnt = 0
                while temp_cnt != uniqu_val:
                    temp_d = np.random.choice(cat_val_list,
                                              size=len(df),
                                              p=[1.0/uniqu_val] * uniqu_val)
                    temp_cnt = len(set(temp_d))
                temp_d = pd.DataFrame(temp_d)
                temp_d.columns = [c]
                if i == 0:
                    x_cat_d = temp_d
                else:
                    x_cat_d = x_cat_d.join(temp_d)
            df_sample = x_cat_d
        if not x_num.empty:
            for i, c in enumerate(x_num.columns):
                temp_d = np.random.uniform(low=min(df[c]),
                                           high=max(df[c]),
                                           size=len(df))
                temp_d = pd.DataFrame(temp_d)
                temp_d.columns = [c]
                if i == 0:
                    x_cont_d = temp_d
                else:
                    x_cont_d = x_cont_d.join(temp_d)
            if not x_cat.empty:
                df_sample = df_sample.join(x_cont_d)
            else:
                df_sample = x_cont_d
        return df_sample

    def gap(self,
            df: pd.DataFrame,
            stop: str = "globalMax",
            n_trial: int = 10
            ) -> pd.DataFrame:
        """
        Gap statistic module.

        Parameters
        ----------
        :df: pandas.DataFrame
            An input pandas dataframe containing all clustering variables

        :stop: str, `optional`, `default:` ``globalMax``

            Allowed methods are::

                oneSE: One standard error as per Tibshirani et al
                firstMax: Local gap statistic maxima
                globalMax: Global gap statistic maxima

        :n_trial: int, `optional`, `default:` ``10``

            Number of bootstrap samples

        """
        self.df_ip = copy.copy(df)
        df = df[self.x_var]
        self.clus_ip = pd.get_dummies(df, drop_first=True).values
        self.clus_ip = self.clus_ip.astype('float64')
        self.max_clus = min(self.max_clus, len(df.drop_duplicates()))
        self.clus_ip = scale(self.clus_ip)
        cluster = {}
        gap = np.zeros(self.max_clus)
        s = np.zeros(self.max_clus)
        for k in range(self.max_clus):
            cluster["kclusters"] = KMeans(n_clusters=(k+1))
            cluster["labels"] = cluster["kclusters"].fit_predict(self.clus_ip)
            cluster["k_compact"] = np.log(self._Compact(self.clus_ip,
                                                        cluster["kclusters"]
                                                        .cluster_centers_,
                                                        cluster["labels"]))
            # Create random data for comparison
            cluster["trial_compact"] = np.zeros(n_trial)
            for trial in range(n_trial):
                samples = pd.get_dummies(self._NRef(), drop_first=True).values
                samples = samples.astype('float64')
                samples = scale(samples)
                sample_labels = cluster["kclusters"].fit_predict(samples)
                cluster["trial_compact"][trial] = \
                    np.log(self._Compact(samples,
                                         cluster["kclusters"].cluster_centers_,
                                         sample_labels))
            gap[k] = sum(cluster["trial_compact"] -
                         cluster["k_compact"]) ** 2 / n_trial
            # Compute standard deviation of the clustering accuracy over
            # multiple trials
            sd = np.sqrt(sum((cluster["trial_compact"] -
                              (sum(cluster["trial_compact"]) /
                               n_trial)) ** 2) /
                         n_trial)
            s[k] = np.sqrt(1 + 1/n_trial) * sd
            # Stopping criteria
            if stop == "oneSE":
                if k > 1 and gap[k-1] >= gap[k] - s[k]:
                    self.gap_val = pd.DataFrame({"Gap": gap, "SD": s})
                    return cluster["labels"]
            elif stop == "firstMax":
                if k > 1 and gap[k] - gap[k-1] < gap[k-1] - gap[k-2]:
                    self.gap_val = pd.DataFrame({"Gap": gap, "SD": s})
                    tmp = gap
                    tmp = pd.DataFrame(tmp)
                    tmp[1] = tmp[0] - tmp[0].shift(1)
                    tmp = tmp.replace([np.inf, -np.inf], np.nan)
                    tmp = tmp.fillna(0)
                    cluster["opt_k"] = tmp[tmp[1] == max(tmp[1])]\
                        .index.values[0] + 1
                    cluster["kclusters"] = KMeans(n_clusters=cluster["opt_k"])
                    cluster["labels"] = cluster["kclusters"]\
                        .fit_predict(self.clus_ip)
                    return cluster["labels"]
        if stop == "globalMax":
            self.gap_val = pd.DataFrame({"Gap": gap, "SD": s})
            cluster["opt_k"] = self.gap_val[self.gap_val["Gap"] ==
                                            max(self.gap_val.iloc[:, 0])]\
                .index.values[0] + 1
            cluster["kclusters"] = KMeans(n_clusters=cluster["opt_k"])
            cluster["labels"] = cluster["kclusters"].fit_predict(self.clus_ip)
        self.df_op = copy.copy(self.df_ip)
        self.df_op["cluster"] = pd.Series(cluster["labels"])
        self.df_op["cluster"] = self.df_op["cluster"] + 1
        return self.df_op


@dataclass
class Knn():
    """KNN module for classification and regression with grid search cv.

    Parameters
    ----------
    :df: pandas.DataFrame

        Pandas dataframe containing the `y_var` and `x_var`

    :y_var: str

        Target variable

    :x_var: list

        List containing independant variables

    :method: str, `optional`, `default:` ``classify``

        Can be either `classify` or `regression`

    """

    def __init__(self,
                 df: pd.DataFrame,
                 y_var: str,
                 x_var: List[str],
                 method: str = "classify"):
        """Initialize variables for module ``Knn``."""
        self.df = df.reset_index(drop=True)
        self.y_var = y_var
        self.x_var = x_var
        self.method = method
        self.model = None
        self._pre_process()

    def _pre_process(self):
        """Pre-process the data, one hot encoding and scaling."""
        df_ip_x = pd.get_dummies(self.df[self.x_var])
        self.x_var = list(df_ip_x.columns)
        df_ip_x = pd.DataFrame(scale(df_ip_x))
        df_ip_x.columns = self.x_var
        self.df = self.df[[self.y_var]].join(df_ip_x)

    def fit(self,
            grid_params: Optional[Dict[str,
                                       List[Union[str, int]]]] = None,
            n_fold: int = 5
            ) -> Dict[str, Any]:
        """Fit KNN model."""
        max_k = max(int(len(self.df)/(n_fold * 2)), 1)
        if grid_params is None:
            grid_params = {"n_neighbors": list(range(1, max_k, 2)),
                           "weights": ["uniform", "distance"],
                           "metric": ["euclidean", "manhattan"]}
        if self.method == "classify":
            gs = GridSearchCV(sn.KNeighborsClassifier(),
                              grid_params,
                              verbose=0,
                              cv=n_fold,
                              n_jobs=1)
        elif self.method == "regression":
            gs = GridSearchCV(sn.KNeighborsRegressor(),
                              grid_params,
                              verbose=0,
                              cv=n_fold,
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


@dataclass
class RandomForest():
    """Random forest module for both classification and regression.

    Parameters
    ----------
    :df_ip: pandas.DataFrame

        Pandas dataframe containing `y_var` and `x_var` variables.

    :y_var: str

        Dependant variable

    :x_var: list

        List containing independant variables

    :method: str

        Available options are: `classify` and `regression`

    """

    def __init__(self,
                 df: pd.DataFrame,
                 y_var: str,
                 x_var: List[str],
                 method: str = "classify"):
        """Initialize variables for module ``RandomForest``."""
        self.df = df.reset_index(drop=True)
        self.y_var = y_var
        self.x_var = x_var
        self.method = method
        self.model = None
        self._pre_process()

    def _pre_process(self):
        """Pre-process the data."""
        df_ip_x = self.df[self.x_var]
        df_ip_x = pd.get_dummies(df_ip_x)
        self.x_var = list(df_ip_x.columns)
        self.df = self.df[[self.y_var]].join(df_ip_x)

    def fit(self,
            grid_param: Optional[Dict[str, Union[bool, int, str]]] = None,
            n_jobs: int = -1,
            verbose: int = 0,
            seed: int = 123456789
            ) -> Dict[str, Union[str, int, bool]]:
        """Fit random forest model.

        Parameters
        ----------
        :grid_param: dict, `optional`, default: `None`

            Dictionary containing::

                bootstrap: [True, False]
                max_depth: [2, 10, 20]
                n_estimators: [500, 1000]
                max_features: ["sqrt", "auto"]
                min_samples_leaf: [1, 5]

        :n_jobs: int, `optional`, default: `-1`

            Number of cores to be used for parallelization

        :verbose: int, `optional`, default: `0`

            Verbose level, ranging from 0 to 2 where 0 is for silent operation.

        :seed: int, `optional`, default: `123456789`

            Seed to reproduce results

        Note
        ----
        Grid search parameters::

            max_depth: 1 to number of features (p)
            n_estimators: As high as possible (1000)
            max_features: sqrt(p) - classify -- p/3 - regression
            min_samples_leaf: 2 - classify -- 5 - regression

        """
        if grid_param is None:
            grid_param = {"bootstrap": [True],
                          "max_depth": list(range(1, len(self.x_var))),
                          "n_estimators": [1000]}
            if self.method == "classify":
                grid_param["max_features"] = ["sqrt"]
                grid_param["min_samples_leaf"] = [2]
            elif self.method == "regression":
                grid_param["max_features"] = [int(len(self.x_var) / 3)]
                grid_param["min_samples_leaf"] = [5]
        if self.method == "classify":
            tmp_model = rf.RandomForestClassifier(oob_score=True,
                                                  random_state=seed)
        elif self.method == "regression":
            tmp_model = rf.RandomForestRegressor(oob_score=True,
                                                 random_state=seed)
        grid = GridSearchCV(estimator=tmp_model,
                            param_grid=grid_param,
                            n_jobs=n_jobs,
                            verbose=verbose,
                            cv=3,
                            return_train_score=True)
        grid.fit(X=self.df[self.x_var], y=self.df[self.y_var])
        max_d = grid.best_params_.get("max_depth")
        max_f = grid.best_params_.get("max_features")
        msf = grid.best_params_.get("min_samples_leaf")
        n_est = grid.best_params_.get("n_estimators")
        if self.method == "classify":
            self.model = rf.RandomForestClassifier(oob_score=True,
                                                   bootstrap=True,
                                                   random_state=seed,
                                                   max_depth=max_d,
                                                   max_features=max_f,
                                                   min_samples_leaf=msf,
                                                   n_estimators=n_est,
                                                   n_jobs=n_jobs)
        elif self.method == "regression":
            self.model = rf.RandomForestRegressor(oob_score=True,
                                                  bootstrap=True,
                                                  random_state=seed,
                                                  max_depth=max_d,
                                                  max_features=max_f,
                                                  min_samples_leaf=msf,
                                                  n_estimators=n_est,
                                                  n_jobs=n_jobs)
        self.model.fit(X=self.df[self.x_var], y=self.df[self.y_var])
        return grid.best_params_

    def predict(self, x_pred: pd.DataFrame) -> pd.DataFrame:
        """Predict values."""
        return self.model.predict(pd.get_dummies(x_pred))


@dataclass
class XGBoost():
    """XGBoost module for time series data.

    Parameters
    ----------
    :y_train: pandas.core.series.Series

        Pandas series containing training data containing the `y_var`
        variable.

    :x_train: pandas.core.frame.DataFrame

        Pandas dataframe containing training data containing the `x_var`
        variable.

    :y_test: pandas.core.series.Series

        Pandas series containing testing data containing the `y_var`
        variable.

    :x_test: pandas.core.frame.DataFrame

        Pandas dataframe containing testing data containing the `x_var`
        variable.

    :opts: dict, `optional`, default: `None`

        Dictionary containing::

            seed: 123456789
            n_jobs: -1
            k_fold: 10
            method: "time_series" or "normal"
            n_iter: 10

    :params: dict, `optional`, default: `None`

        Dictionary containing::

            n_estimators: [1000]
            learning_rate: [0.002, 0.005]
            subsample: [0.5, 0.7]
            colsample_bytree: [0.1, 0.2]
            min_child_weight: [1, 2]
            max_depth: [2, 3, 4]
            objective: ["reg:squarederror"]
            gamma: [0, 1, 2]

    """

    def __init__(self,
                 y_train: pd.Series,
                 x_train: pd.DataFrame,
                 y_test: pd.Series,
                 x_test: pd.DataFrame,
                 opts: Optional[Dict[str, Union[str, int]]] = None,
                 params: Optional[Dict[str,
                                       List[Union[str,
                                                  int,
                                                  float]]]] = None):
        """Initialize variables for module ``XGBoost``."""
        self.y_train = y_train
        self.x_train = x_train
        self.y_test = y_test
        self.x_test = x_test
        x_vars = list(self.x_train.columns)
        if opts is None:
            self.opts = {"seed": 123456789,
                         "n_jobs": -1,
                         "k_fold": 10,
                         "method": "time_series",
                         "n_iter": 10}
        else:
            self.opts = opts
        if params is None:
            max_depth = list(range(3, max(int(len(x_vars)/3), 4)))
            self.params = {"n_estimators": [1000],
                           "learning_rate": [i/1000 for i in range(2, 11)],
                           "subsample": [i/10 for i in range(5, 10)],
                           "colsample_bytree": [i/10 for i in range(1, 11)],
                           "min_child_weight": list(range(1, 11)),
                           "max_depth": max_depth,
                           "objective": ["reg:squarederror"],
                           "gamma": list(range(0, 21)),
                           }
        else:
            self.params = params
        self.xgb_model = None
        self.one_se = True

    def _best_param(self,
                    obj: Dict[str,
                              Union[List[Union[float,
                                               int,
                                               Dict[str,
                                                    Union[int,
                                                          str,
                                                          float]]]]]]
                    ) -> Tuple[Dict[str, Union[float, int, str]],
                               List[List[float]]]:
        """Determine optimal hyper-parameters.

        Parameters
        ----------
        :obj: XGB object

            A XGBoost object containing the cross validation results
            (xgb.cv_results_).

        Returns
        -------
        :Optimal hyper-parameters: dict

        :One standard error: list

        """
        mean_test_score = obj["mean_test_score"]
        std_test_score = obj["std_test_score"]
        mean_test_score = list((-1 * mean_test_score))
        std_test_score = list(((std_test_score) ** 2) /
                              (self.opts["k_fold"] ** 0.5))
        one_se = []
        for i in range(len(obj["params"])):
            est_rs = xgb.XGBRegressor(n_jobs=self.opts["n_jobs"],
                                      verbosity=0,
                                      silent=True,
                                      random_state=self.opts["seed"],
                                      seed=self.opts["seed"],
                                      **obj["params"][i])
            est_rs.fit(self.x_train, self.y_train)
            y_tmp = list(est_rs.predict(self.x_test))
            one_se.append([mean_test_score[i], std_test_score[i],
                           mean_test_score[i] + std_test_score[i],
                           max(mean_test_score[i] - std_test_score[i], 0),
                           metrics.mse(self.y_test.tolist(),
                                       y_tmp)])
        one_se_op = [i[4] if (i[2] >= i[4] >= i[3]) else np.Inf
                     for i in one_se]
        if min(one_se_op) == np.Inf:
            return -1
        opt_param = one_se_op.index(min(one_se_op))
        return (obj["params"][opt_param], one_se)

    def fit(self):
        """Fit XGBoost model.

        Parameters
        ----------
        :force: bool, `optional`, `default:` ``False``

            When True, the module will return best model irrespective of one
            standard error rule.

        Returns
        -------
        :flag: bool

            True if model exists, False otherwise

        """
        est_xgb = xgb.XGBRegressor(n_jobs=self.opts["n_jobs"],
                                   verbosity=0,
                                   silent=True,
                                   random_state=self.opts["seed"],
                                   seed=self.opts["seed"])
        if self.opts["method"] == "time_series":
            self.opts["fold"] = tssplit(n_splits=self.opts["k_fold"])\
                .split(X=self.x_train, y=self.y_train)
        elif self.opts["method"] == "normal":
            self.opts["fold"] = self.opts["k_fold"]
        rs_xgb = RandomizedSearchCV(estimator=est_xgb,
                                    param_distributions=self.params,
                                    n_jobs=self.opts["n_jobs"],
                                    cv=self.opts["fold"],
                                    verbose=0,
                                    random_state=self.opts["seed"],
                                    n_iter=self.opts["n_iter"],
                                    scoring="neg_mean_squared_error")
        rs_xgb.fit(self.x_train, self.y_train)
        tmp = self._best_param(rs_xgb.cv_results_)
        if tmp == -1:
            self.one_se = False
            fit_param = rs_xgb.best_params_
        else:
            fit_param = self._best_param(rs_xgb.cv_results_)[0]
        self.xgb_model = xgb.XGBRegressor(n_jobs=self.opts["n_jobs"],
                                          verbosity=0,
                                          silent=True,
                                          random_state=self.opts["seed"],
                                          seed=self.opts["seed"],
                                          **fit_param)
        self.xgb_model.fit(self.x_train.append(self.x_test),
                           self.y_train.append(self.y_test))

    def predict(self, x_pred: pd.DataFrame) -> List[Union[int, float]]:
        """Predict using the XGBoost model built.

        Parameters
        ----------
        :x_pred: pandas.core.frame.DataFrame

            A pandas dataframe containing all independant variables with which
            dependant variable needs to be predicted.

        Returns
        -------
        list

            Target predicted

        """
        return list(self.xgb_model.predict(x_pred))
