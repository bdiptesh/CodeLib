"""
Unit tests for stat.py module.

Author
------
::

    Author: Diptesh.Basak
    Date: Jun 14, 2019
    License: BSD 3-Clause
"""

# pylint: disable=invalid-name

import unittest
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split as split

from __init__ import Model
from __init__ import Cluster
from __init__ import Knn
from __init__ import RandomForest
from __init__ import path


# =============================================================================
# --- User defined functions
# =============================================================================


class Test_GLMnet(unittest.TestCase):
    """
    Test suite for GLMnet module
    """

    def setUp(self):
        """Setup for module ``Test_GLMnet``."""

    def test_known_equation(self):
        """ Test a known equation for GLMnet
        """
        df_ip = pd.read_csv(path + "/data/input/test_glmnet.csv")
        mod = Model(test_perc=0.25)
        op = mod.cv_glmnet(df=df_ip,
                           y_var="y",
                           x_var=["x1", "x2", "x3"])[0]
        self.assertEqual(np.round(op.get('intercept'), 0), 102.0)
        self.assertEqual(np.round(op.get('coef')[0], 0), 2.0)
        self.assertEqual(np.round(op.get('coef')[1], 0), 3.0)
        self.assertEqual(np.round(op.get('coef')[2], 0), 0.0)


class Test_Cluster(unittest.TestCase):
    """
    Test suite for clustering
    """

    def setUp(self):
        """Setup for module ``Test_Cluster``."""

    def test_categorical(self):
        """ Test clustering with categorical data
        """
        df_ip = pd.read_csv(path + "/data/input/store.csv")
        coln = ["x1"]
        df_ip = df_ip[coln]
        tmp = Cluster(max_clus=20, x_var=coln)
        tmp_op = tmp.gap(df=df_ip)
        self.assertEqual(max(tmp_op["cluster"]), 4)

    def test_continuous(self):
        """ Test clustering with continuous data
        """
        df_ip = pd.read_csv(path + "/data/input/store.csv")
        coln = ["x6"]
        df_ip = df_ip[coln]
        tmp = Cluster(max_clus=10, x_var=coln)
        tmp_op = tmp.gap(df=df_ip)
        self.assertEqual(max(tmp_op["cluster"]), 10)

    def test_mixed(self):
        """ Test clustering with categorical and continuous data
        """
        df_ip = pd.read_csv(path + "/data/input/store.csv")
        coln = ["x1", "x3"]
        df_ip = df_ip[coln]
        tmp = Cluster(max_clus=20, x_var=coln)
        tmp_op = tmp.gap(df=df_ip)
        self.assertEqual(max(tmp_op["cluster"]), 5)


class Test_RandomForest(unittest.TestCase):
    """
    Test suite for random forest.
    """

    def setUp(self):
        """Setup for module ``Test_RandomForest``."""

    def test_rf_class(self):
        """ Test random forest classification.
        """
        df_ip = pd.read_csv(path + "/data/input/iris.csv")
        train, test = split(df_ip,
                            stratify=df_ip["y"],
                            test_size=0.1,
                            random_state=42)
        mod = RandomForest(df=train,
                           y_var="y",
                           x_var=["x1", "x2", "x3", "x4"],
                           method="classify")
        mod.fit()
        y_hat = list(mod.predict(test[["x1", "x2", "x3", "x4"]]))
        y = test["y"].values.tolist()
        acc = round(len([i for i, j in zip(y, y_hat) if i == j]) / len(y), 2)
        self.assertGreaterEqual(acc, 0.93)

    def test_rf_reg(self):
        """ Test random forest regression.
        """
        df_ip = pd.read_csv(path + "/data/input/iris.csv")
        train, test = split(df_ip,
                            stratify=df_ip["y"],
                            test_size=0.1,
                            random_state=42)
        mod = RandomForest(df=train,
                           y_var="y",
                           x_var=["x1", "x2", "x3", "x4"],
                           method="classify")
        mod.fit()
        y_hat = list(mod.predict(test[["x1", "x2", "x3", "x4"]]))
        y_hat = [int(round(i)) for i in y_hat]
        y = test["y"].values.tolist()
        acc = round(len([i for i, j in zip(y, y_hat) if i == j]) / len(y), 2)
        self.assertGreaterEqual(acc, 0.87)


class Test_Knn(unittest.TestCase):
    """
    Test suite for KNN.
    """

    def setUp(self):
        """Setup for module ``Test_Knn``."""

    def test_knn_class(self):
        """ Test KNN classification.
        """
        df_ip = pd.read_csv(path + "/data/input/iris.csv")
        df_ip = df_ip[["y", "x1", "x2"]]
        df_train, df_test = split(df_ip,
                                  stratify=df_ip["y"],
                                  test_size=0.1,
                                  random_state=42)
        mod = Knn(df_train, "y", ["x1", "x2"], method="classify")
        mod.fit()
        y_hat = mod.predict(df_test[["x1", "x2"]]).tolist()
        y = df_test["y"].values.tolist()
        acc = round(len([i for i, j in zip(y, y_hat) if i == j]) / len(y), 2)
        self.assertGreaterEqual(acc, 0.93)

    def test_knn_reg(self):
        """ Test KNN regression.
        """
        df_ip = pd.read_csv(path + "/data/input/iris.csv")
        df_ip = df_ip[["y", "x1", "x2"]]
        df_train, df_test = split(df_ip,
                                  stratify=df_ip["y"],
                                  test_size=0.1,
                                  random_state=42)
        mod = Knn(df_train, "y", ["x1", "x2"], method="regression")
        mod.fit()
        y_hat = mod.predict(df_test[["x1", "x2"]]).tolist()
        y = df_test["y"].values.tolist()
        acc = round(len([i for i, j in zip(y, y_hat) if i == j]) / len(y), 2)
        self.assertGreaterEqual(acc, 0.87)


# =============================================================================
# --- Main
# =============================================================================

if __name__ == '__main__':
    unittest.main()
