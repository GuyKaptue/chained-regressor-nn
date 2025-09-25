"""
utils.py
========

Utility functions and default regressor definitions for the ChainedRegressorNN pipeline.

This module provides:
---------------------
1. safe_hstack_hybrid — a robust horizontal stacking function that preserves
   sparsity when possible and ensures dimension safety.
2. DEFAULT_REGRESSORS — a dictionary of pre‑configured scikit‑learn regressors,
   with optional inclusion of XGBoost, LightGBM, and CatBoost if installed.

Author
------
Guy Kaptue
"""

from __future__ import annotations

import numpy as np # type:ignore
from scipy.sparse import issparse, hstack as sparse_hstack, csr_matrix # type:ignore
from typing import Any, Union

# Core regressors from scikit-learn
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet # type:ignore
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor # type:ignore
from sklearn.svm import SVR # type:ignore
from sklearn.neighbors import KNeighborsRegressor # type:ignore

# Optional regressors from external libraries
from xgboost import XGBRegressor # type:ignore
from lightgbm import LGBMRegressor # type:ignore
from catboost import CatBoostRegressor # type:ignore

def get_default_regressors() -> dict[str, Any]:
    """
    Return a dictionary of default regressors, including optional ones if installed.
    """
    defaults: dict[str, Any] = {
        "linear": LinearRegression(),
        "ridge": Ridge(random_state=42),
        "lasso": Lasso(random_state=42),
        "elastic_net": ElasticNet(random_state=42),
        "random_forest": RandomForestRegressor(random_state=42),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
        "svr": SVR(),
        "knn": KNeighborsRegressor(),
    }

    if XGBRegressor is not None:
        defaults["xgboost"] = XGBRegressor(random_state=42)
    if LGBMRegressor is not None:
        defaults["lightgbm"] = LGBMRegressor(random_state=42)
    if CatBoostRegressor is not None:
        defaults["catboost"] = CatBoostRegressor(random_state=42, verbose=0)

    return defaults



def safe_hstack_hybrid(
    a: Union[np.ndarray, Any],
    b: Union[np.ndarray, Any],
    force_dense: bool = False
) -> Union[np.ndarray, Any]:
    """
    Horizontally stack features with dimension safety and sparsity preservation.

    Parameters
    ----------
    a : array-like or sparse matrix
        First array to stack.
    b : array-like or sparse matrix
        Second array to stack.
    force_dense : bool, default=False
        If True, converts both arrays to dense before stacking.

    Returns
    -------
    array-like or sparse matrix
        Horizontally stacked arrays, preserving sparsity unless `force_dense` is True.
    """
    if b.ndim == 1:
        b = b.reshape(-1, 1)

    if force_dense:
        if issparse(a):
            a = a.toarray()
        if issparse(b):
            b = b.toarray()
        return np.hstack([a, b])

    if issparse(a) and issparse(b):
        return sparse_hstack([a, b])
    if issparse(a) and not issparse(b):
        b = csr_matrix(b)
        return sparse_hstack([a, b])
    if not issparse(a) and issparse(b):
        a = csr_matrix(a)
        return sparse_hstack([a, b])

    return np.hstack([a, b])


# Default regressors configuration
DEFAULT_REGRESSORS: dict[str, Any] = get_default_regressors()
