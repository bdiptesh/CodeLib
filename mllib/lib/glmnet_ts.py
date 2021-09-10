#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 15:25:51 2021

@author: madhu
"""

from typing import List, Dict

import pandas as pd
import numpy as np

from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split as split
from sklearn.model_selection import TimeSeriesSplit as ts_split

import metrics


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


df_ip = pd.read_csv(
    "/media/ph33r/Data/Project/mllib/GitHub/data/input/test_timeseries.csv")

y_var = ["y"]
x_var = ["x1", "x2"]

param = {}
param["a_inc"] = 0.05
param["k_fold"] = 5
param["test_perc"] = 0.2
param["n_jobs"] = -1
param["seed"] = 1
param["l1_range"] = list(np.round(np.arange(0.0001, 1.01, param["a_inc"]), 2))


df_ip = create_lag_vars(df_ip, y_var, x_var, "week")
# modify create lag function to get lag list
lag_var = [52, 26, 13, 6, 4, 3, 2, 1]
x_var = list(df_ip.columns)
x_var.remove(y_var[0])

# Use len?
max_epoch = df_ip.index.max() + 1

# For prediction
df_pred_data = df_ip[y_var]

# Use iloc
df_train = df_ip[df_ip.index <= max_epoch * (1-param["test_perc"])]
df_test = df_ip[df_ip.index > (max_epoch) * (1-param["test_perc"])]

train_x = df_train[x_var]
train_y = df_train[y_var]

# Should it not be df_test?
test_x = df_train[x_var]
test_y = df_train[y_var]

test_x = df_test[x_var]
test_y = df_test[y_var]

param["k_fold"] = ts_split(n_splits=param["k_fold"])
param["k_fold"] = param["k_fold"].split(X=train_y)


mod = ElasticNetCV(l1_ratio=param["l1_range"],
                   fit_intercept=True,
                   alphas=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
                           1.0, 10.0, 100.0],
                   normalize=True,
                   cv=param["k_fold"],
                   n_jobs=param["n_jobs"],
                   random_state=param["seed"])

mod.fit(train_x, train_y.values.ravel())

opt = {"alpha": mod.l1_ratio_,
       "lambda": mod.alpha_,
       "intercept": mod.intercept_,
       "coef": mod.coef_,
       "train_v": mod.score(train_x, train_y),
       "test_v": mod.score(test_x, test_y)}
model = mod
opt = opt


# Prediction
df_predict = df_test.copy(deep=True)
df_predict = df_ip.copy(deep=True)

# reset index
df_predict = df_predict.reset_index(drop=True)
df_predict = df_predict[["x1", "x2"]]
df_predict["y"] = -1

# Is there a way to improve this?
for i in range(0, len(df_test)):
    # for i in range(0, len(df_ip)):
    df_pred = df_predict[df_predict.index == i].reset_index(drop=True)
    df_pred = df_pred[["x1", "x2"]]
    df_pred_x = pd.DataFrame(
        {"lag_"+str(lag_var[0]): df_pred_data.iloc[len(df_pred_data)-lag_var[0]]})
    for j in range(1, len(lag_var)):
        df_tmp = pd.DataFrame(
            {"lag_"+str(lag_var[j]): df_pred_data.iloc[len(df_pred_data)-lag_var[j]]})
        df_pred_x = df_pred_x.join(df_tmp)
    df_pred_x = df_pred_x.reset_index(drop=True)
    df_pred_x = df_pred_x.join(df_pred)
    y_hat = model.predict(df_pred_x)
    df_tmp = pd.DataFrame()
    df_tmp['y'] = y_hat
    df_pred_data = df_pred_data.append(df_tmp).reset_index(drop=True)
    df_predict["y"][i] = y_hat


y = list(df_ip["y"])
y_hat = list(df_predict["y"])
model_summary = {"rsq": metrics.rsq(y, y_hat),
                 "mae": metrics.mae(y, y_hat),
                 "mape": metrics.mape(y, y_hat),
                 "rmse": metrics.rmse(y, y_hat)}
model_summary["mse"] = model_summary["rmse"] ** 2
model_summary
