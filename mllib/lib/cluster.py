"""
Clustering module.

Objective:
    - Determine optimal number of clusters using
      `Gap statistic <https://web.stanford.edu/~hastie/Papers/gap.pdf>`_.

Credits
-------
::

    Authors:
        - Diptesh

    Date: Sep 05, 2021
"""

# pylint: disable=invalid-name
# pylint: disable=R0902,R0903,R0913,R0914

from typing import List

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

# =============================================================================
# --- DO NOT CHANGE ANYTHING FROM HERE
# =============================================================================


class Cluster():
    """
    Clustering module.

    Objective:
        - Determine optimal number of clusters using
          `Gap statistic <https://web.stanford.edu/~hastie/Papers/gap.pdf>`_.

    Parameters
    ----------
    df : pd.DataFrame

        Dataframe containing all clustering variables i.e. `x_var`.

    x_var : List[str]

        List of clustering variables.

    max_cluster : int, optional

        Maximum number of clusters. The default is 20.

    nrefs : int, optional

        Number of random references to be created. The default is 20.

    seed : int, optional

        Random seed. The default is 1.

    method : str, optional

        Stopping criterion (`one_se` or `gap_max`). The default is `one_se`.

    """

    def __init__(self,
                 df: pd.DataFrame,
                 x_var: List[str],
                 max_cluster: int = 20,
                 nrefs: int = 20,
                 seed: int = 1,
                 method: str = "one_se"):
        """Initialize variables for module ``Cluster``."""
        self.df = df
        self.x_var = x_var
        self.max_cluster = max_cluster
        self.nrefs = nrefs
        self.seed = seed
        self.clus_op: pd.DataFrame = None
        self.optimal_k: int = None
        self.df_gap: pd.DataFrame = None
        self.df = self.df[self.x_var]
        self.max_cluster = min(self.max_cluster,
                               len(self.df.drop_duplicates()))
        x_cat = self.df.select_dtypes(include=['object', 'bool'])
        if not x_cat.empty:
            self.method = "gap_max"
        else:
            self.method = method

    def _nref(self):
        """Create random reference data."""
        df = self.df
        x_cat = df.select_dtypes(include=['object', 'bool'])
        x_num = df.select_dtypes(include=['int', 'float64'])
        if not x_cat.empty:
            for _, cat_col in enumerate(x_cat.columns):
                cat_val_list = df[cat_col].unique()
                uniqu_val = len(cat_val_list)
                temp_cnt = 0
                while temp_cnt != uniqu_val:
                    temp_d = np.random.choice(cat_val_list,
                                              size=len(df),
                                              p=[1.0/uniqu_val] * uniqu_val)
                    temp_cnt = len(set(temp_d))
                temp_d = pd.DataFrame(temp_d)
                temp_d.columns = [cat_col]
                if _ == 0:
                    x_cat_d = temp_d
                else:
                    x_cat_d = x_cat_d.join(temp_d)
            df_sample = x_cat_d
        if not x_num.empty:
            for _, num_col in enumerate(x_num.columns):
                temp_d = np.random.uniform(low=min(df[num_col]),
                                           high=max(df[num_col]),
                                           size=len(df))
                temp_d = pd.DataFrame(temp_d)
                temp_d.columns = [num_col]
                if _ == 0:
                    x_cont_d = temp_d
                else:
                    x_cont_d = x_cont_d.join(temp_d)
            if not x_cat.empty:
                df_sample = df_sample.join(x_cont_d)
            else:
                df_sample = x_cont_d
        df_sample = pd.get_dummies(data=df_sample, drop_first=True)
        df_sample = pd.DataFrame(scale(df_sample))
        return df_sample

    def opt_k(self):
        """Compute optimal number of clusters using gap statistic.

        Returns
        -------
        pd.DataFrame

            pandas dataframe containing::

                x_var
                cluster

        """
        df = self.df
        # One hot encoding of categorical variables
        df_clus = pd.get_dummies(data=df, drop_first=True)
        # Scale the data
        df_clus_ip = pd.DataFrame(scale(df_clus))
        # Create arrays for gap and sk
        gaps = np.zeros(self.max_cluster)
        sks = np.zeros(self.max_cluster)
        # Create results dataframe
        df_result = pd.DataFrame({"cluster": [], "gap": [], "sk": []})
        # Create new random reference set
        dict_nref = dict()
        for i in range(self.nrefs):
            dict_nref[i] = self._nref()
        # Compute gap statistic
        for gap_index, k in enumerate(range(1, self.max_cluster + 1)):
            # Holder for reference dispersion results
            ref_disps = np.zeros(self.nrefs)
            # For n references, generate random sample and perform kmeans
            # getting resulting dispersion of each loop
            for i in range(self.nrefs):
                # Create new random reference set
                random_ref = dict_nref[i]
                # Fit to it
                km = KMeans(k, random_state=self.seed)
                km.fit(random_ref)
                ref_disp = km.inertia_
                ref_disps[i] = ref_disp
            # Fit cluster to original data and create dispersion
            km = KMeans(k, random_state=self.seed)
            km.fit(df_clus_ip)
            orig_disp = km.inertia_
            # Calculate gap statistic
            gap = np.inf
            if orig_disp > 0.0:
                gap = np.mean(np.log(ref_disps)) - np.log(orig_disp)
            # Compute standard error
            sk = 0.0
            if sum(ref_disps) != 0.0:
                sdk = np.std(np.log(ref_disps))
                sk = sdk * np.sqrt(1.0 + 1.0 / self.nrefs)
            # Assign this loop's gap statistic and sk to gaps and sks
            gaps[gap_index] = gap
            sks[gap_index] = sk
            df_result = df_result.append({"cluster": k,
                                          "gap": gap,
                                          "sk": sk},
                                         ignore_index=True)
            # Stopping criteria
            if self.method == "one_se":
                if k > 1 and gaps[gap_index-1] >= gap - sk:
                    opt_k = k-1
                    km = KMeans(opt_k, random_state=self.seed)
                    km.fit(df_clus_ip)
                    clus_op = km.labels_
                    break
            opt_k = np.argmax(gaps) + 1
            km = KMeans(opt_k, random_state=self.seed)
            km.fit(df_clus_ip)
            clus_op = km.labels_
        self.df_gap = df_result
        self.optimal_k = opt_k
        self.clus_op = pd.concat([self.df,
                                  pd.DataFrame(data=clus_op,
                                               columns=["cluster"])],
                                 axis=1)
        return self.clus_op
