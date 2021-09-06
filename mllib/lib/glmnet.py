"""
GLMNet module.

Objective:
    - Build
      `GLMNet <https://web.stanford.edu/~hastie/Papers/glmnet.pdf>`_
      model using optimal alpha and lambda

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

# =============================================================================
# --- DO NOT CHANGE ANYTHING FROM HERE
# =============================================================================


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

        Dependant variables.

    x_var : List[str]

        Independant variables.

    timeseries : bool, optional

        Boolean value to indicate time-series inputs (the default is False).

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

    """

    def __init__(self,
                 df: pd.DataFrame,
                 y_var: List[str],
                 x_var: List[str],
                 timeseries: bool = False,
                 param: Dict = None):
        """Initialize variables for module ``GLMNet``."""
        self.df = df
        self.y_var = y_var
        self.x_var = x_var
        if param is None:
            param = {"seed": 1,
                     "a_inc": 0.05,
                     "test_perc": 0.25,
                     "n_jobs": -1,
                     "k_fold": 10,
                     "lambda_param": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
                                      1.0, 10.0, 100.0]}
        self.param = param
        self.param["l1_range"] = list(np.array(range(5, 105,
                                                     int(self.param["a_inc"]
                                                         * 100.0))) / 100.0)
        self.param["timeseries"] = timeseries

    def fit(self):
        """Fit the best GLMNet model."""

    def predict(self, df_predict: pd.DataFrame) -> pd.DataFrame:
        """Short summary.

        Parameters
        ----------
        df_predict : pd.DataFrame

            Pandas dataframe containing `x_var`.

        Returns
        -------
        pd.DataFrame

            Pandas dataframe containing predicted `y_var` and `x_var`.

        """
