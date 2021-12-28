"""
Time series module.

**Available routines:**

- class ``AutoArima``: Builds time series model using SARIMAX.

Credits
-------
::

    Authors:
        - Diptesh
        - Madhu

    Date: Dec 28, 2021
"""

# pylint: disable=invalid-name
# pylint: disable=wrong-import-position
# pylint: disable=R0902,R0903,W0511

from inspect import getsourcefile
from os.path import abspath

from typing import Tuple, Dict, List, Union
from itertools import product

import re
import sys
import warnings

import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import acf, pacf, adfuller
import statsmodels.api as sm
import scipy.stats as stats

path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+\/)(.+.py)", "\\1", path)
sys.path.insert(0, path)

import metrics  # noqa: F841


class AutoArima():
    """Auto ARIMA time series module.

    Parameters
    ----------
    df: pandas.DataFrame

        Pandas dataframe containing the `y_var` and optinal `x_var`

    y_var: str

        Dependant variable

    x_var: List[str], optional

        Independant variables (the default is None).

    param: dict, optional

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
    >>> mod = AutoArima(df=df_ip,
                        y_var="y",
                        x_var=["cost", "stock_level", "retail_price"])
    >>> df_op = mod.predict(x_predict)

    Approach
    --------

    Determine optimal `d`:

    - Check for stationarity for values of `d`. If data is not stationary
      where `d` equals to `max_d`, raise a warning and continue.

    Determine max `p`, `q`:

    - Compute max `p` from pacf and `q` from acf.

    Determine optimal `p`, `q`:

    - Run a grid search with `max_p`, `d`, `max_q`.
    - Optimal `p`, `d`, `q` corresponds to the model with least AIC.

    """

    def __init__(self,
                 df: pd.DataFrame,
                 y_var: str,
                 x_var: List[str] = None,
                 param: Dict = None
                 ):
        """Initialize variables."""
        self.df = df
        self.y_var = y_var
        self.x_var = x_var
        if param is None:  # pragma: no cover
            param = {"max_p": 20,
                     "max_d": 2,
                     "max_q": 7,
                     "threshold": 0.05}
        self.param = param
        self.opt_pdq = None
        self.aic_val = None
        self.model = None
        self.y_hat = None
        self.model_summary = None
        # TODO: Add decomposition
        # TODO: Add PDQs
        self._opt_pdq(df=self.df[self.y_var], params=self.param)
        self._fit(self.opt_pdq)
        self._compute_metrics()

    def _opt_pdq(self,
                 df: pd.core.series.Series,
                 params: Dict[str, Union[int, float]]
                 ) -> Tuple[int, int, int]:
        """Determine optimal `p`, `d`, `q` values.

        Parameters
        ----------
        df : pandas.core.series.Series

            Pandas series containing the target variable only.

        params : Dict, optional

            Parameters to compute optimal `p`, `d`, `q` values.

        Returns
        -------
        tuple

            Optimal `p`, `d`, `q` values.

        """
        # Determine optimal d
        for d in range(params["max_d"]):  # pragma: no cover
            p_val = adfuller(df, autolag="AIC", maxlag=params["max_p"])[1]
            if p_val <= params["threshold"]:
                break
            if (p_val > params["threshold"]) and (d == params["max_d"]):
                warnings.warn("Maximum value of d reached. Check input data.")
                break
            df = df - df.shift(1)
            df = df.dropna()
        # Determine max p and q
        ts_len = len(df)
        df = pd.DataFrame({"lag": pd.Series(list(range(params["max_p"] + 1))),
                           "acf": pd.Series(acf(df,
                                                nlags=params["max_q"],
                                                fft=False)),
                           "pacf": pd.Series(pacf(df,
                                                  nlags=params["max_p"],
                                                  method='ols'))})
        df["thres_val"] = (np.round(stats.norm.ppf(1 - (params["threshold"]
                                                        / 2)), 2)
                           / ((ts_len - d) ** 0.5))
        df["acf_sig"] = np.where((abs(df['acf']) > df["thres_val"]),
                                 1, 0)
        df["pacf_sig"] = np.where((abs(df['pacf']) > df["thres_val"]),
                                  1, 0)
        max_p = int(max(0, df[df["pacf_sig"] == 1].max()["lag"]))
        max_q = int(max(0, df[df["acf_sig"] == 1].max()["lag"]))
        # Grid search
        pdq_val = list(product(list(range(max_p + 1)),
                               [d],
                               list(range(max_q + 1))))
        aic_val = {}
        for pdq in pdq_val:
            aic_val[pdq] = self._fit(pdq=pdq)
        opt_pdq = min(aic_val, key=aic_val.get)
        self.opt_pdq = opt_pdq
        self.aic_val = aic_val
        return opt_pdq

    def _compute_metrics(self):
        """Compute commonly used metrics to evaluate the model."""
        y = self.df[[self.y_var]].iloc[:, 0].values.tolist()
        if self.x_var is None:
            y_hat = list(self.model.predict(start=1, end=len(self.df)))
        else:
            y_hat = list(self.predict(self.df[self.x_var])["y"].values)
        model_summary = {"rsq": np.round(metrics.rsq(y, y_hat), 3),
                         "mae": np.round(metrics.mae(y, y_hat), 3),
                         "mape": np.round(metrics.mape(y, y_hat), 3),
                         "rmse": np.round(metrics.rmse(y, y_hat), 3)}
        model_summary["mse"] = np.round(model_summary["rmse"] ** 2, 3)
        self.y_hat = y_hat
        self.model_summary = model_summary

    def _fit(self, pdq: Tuple[int, int, int]) -> float:
        """Fit a `SARIMAX` model for a given `p`, `d` and `q` values.

        Parameters
        ----------
        pdq : Tuple[int, int, int]

            Tuple containing `p`, `d` and `q` values.

        Returns
        -------
        float

            AIC for the fitted model.

        """
        if self.x_var is None:
            model = sm.tsa.statespace.SARIMAX(endog=self.df[self.y_var],
                                              order=pdq)
        else:
            model = sm.tsa.statespace.SARIMAX(endog=self.df[self.y_var],
                                              exog=self.df[self.x_var],
                                              order=pdq)
        model_op = model.fit(disp=False, method="powell")
        self.model = model_op
        return model_op.aic

    def predict(self,
                x_predict: pd.DataFrame = None,
                n_interval: int = 1) -> pd.DataFrame:
        """Predict module.

        Parameters
        ----------
        x_predict : pd.DataFrame, optional

            Pandas dataframe containing `x_var` (the default is None).

        n_interval : int, optional

            Number of time period to predict (the default is 1).

        Returns
        -------
        pd.DataFrame

            Pandas dataframe containing `y_var` and `x_var` (optional).

        """
        if self.x_var is None:
            df_pred = pd.DataFrame(self.model.predict(start=len(self.df),
                                                      end=len(self.df)
                                                      + n_interval))
            df_pred = df_pred.iloc[0:len(df_pred) - 1]
            df_pred.columns = [self.y_var]
        else:
            df_pred = pd.DataFrame(self.model.predict(exog=x_predict))
            df_pred = pd.concat([df_pred, self.df[self.x_var]], axis=1)
            df_pred.columns = list(self.y_var) + self.x_var
        return df_pred
