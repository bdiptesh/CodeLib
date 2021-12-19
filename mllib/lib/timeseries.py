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

import itertools
import statsmodels.api as sm

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
                 param: Dict = None,
                 epoch = 'days'):
        """Initialize variables."""
        self.df = df
        self.y_var = y_var
        self.x_var = x_var
        self._check_data()
        if param is None:
            param = {'p': [0, 1], 'd': [0, 1], 'q': [0]}
            if epoch == 'days':
                param['S'] = [7, 30, 365]
            elif epoch == 'month':
                param['S'] = [12]
            else:
                param['S'] = [0]
        self.param = param
        self.best_param = None
        self._check_data()
        self._opt_param()
        self._fit()

    def _check_data(self):
        df_check = self.df.dropna()
        if len(self.df) != len(df_check):
            raise ValueError("Found missing values in input data")
        # TO DO: Check for y_var, x_var, number of observations.

    def _opt_param(self):
        """Determine optimal parameters."""
        # Generate all different combinations of p, q and q triplets
        pdq = list(itertools.product(self.param['p'],
                                     self.param['d'],
                                     self.param['q']))
        # Generate all different combinations of seasonal p, q and q triplets
        pdqs = [(x[0], x[1], x[2], 12) \
                for x in list(itertools.product(self.param['p'],
                                                self.param['d'],
                                                self.param['q']))]
        # Run a grid with pdq and seasonal pdq parameters calculated above and 
        # get the best AIC value
        ans = []
        for comb in pdq:
            for combs in pdqs:
                try:
                    if self.x_var is None:
                        mod = sm.tsa.statespace.SARIMAX(endog = self.df[self.y_var],
                                                        order=comb,
                                                        seasonal_order=combs,
                                                        enforce_stationarity=False,
                                                        enforce_invertibility=False,
                                                        )
                    else:
                        mod = sm.tsa.statespace.SARIMAX(endog = self.df[self.y_var],
                                                        exog = self.df[self.x_var], 
                                                        order=comb,
                                                        seasonal_order=combs,
                                                        enforce_stationarity=False,
                                                        enforce_invertibility=False
                                                        )
                    output = mod.fit(disp=0)
                    ans.append([comb, combs, output.aic])
                    print('ARIMA {} x {}12 : \
                          AIC Calculated ={}'.format(comb, combs, output.aic))
                except:
                    continue
        # Find the parameters with minimal AIC value
        ans_df = pd.DataFrame(ans, columns=['pdq', 'pdqs', 'aic'])
        ans_df = ans_df.loc[ans_df['aic'].idxmin()]
        self.best_param = {'pdq': ans_df[0], 'pdqs': ans_df[1]}
        return
    
    def _fit(self):
        """Fit the model."""
        if self.x_var is None:
            mod = sm.tsa.statespace.SARIMAX(endog = self.df[self.y_var],
                                            order=self.best_param['pdq'],
                                            seasonal_order=self.best_param['pdqs'],
                                            enforce_stationarity=False,
                                            enforce_invertibility=False
                                            )
        else:
            mod = sm.tsa.statespace.SARIMAX(endog = self.df[self.y_var],
                                            exog = self.df[self.x_var], 
                                            order=self.best_param['pdq'],
                                            seasonal_order=self.best_param['pdqs'],
                                            enforce_stationarity=False,
                                            enforce_invertibility=False
                                            )
        self.model = mod.fit(disp=0)
        return

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
        if self.x_var is None:
            df_pred = pd.DataFrame(\
                            self.model.predict(start = len(self.df) + 1,
                                               end=len(self.df) + n_interval))
        else:
            df_pred = pd.DataFrame(self.model.predict(exog = x_predict))
        return df_pred
