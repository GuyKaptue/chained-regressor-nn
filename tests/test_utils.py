import numpy as np # type:ignore
from typing import Any
import pytest # type:ignore
from sklearn.linear_model import LinearRegression # type:ignore
from scipy.sparse import csr_matrix, issparse # type:ignore

import chainedregressornn.utils as utils

DEFAULT_REGRESSORS = utils.DEFAULT_REGRESSORS
# -------------------------------------------------------------------
# safe_hstack_hybrid tests
# -------------------------------------------------------------------

def test_safe_hstack_dense_dense():
    a = np.ones((2, 2))
    b = np.zeros((2, 1))
    result = utils.safe_hstack_hybrid(a, b)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3)
    assert np.allclose(result[:, :2], 1.0)
    assert np.allclose(result[:, 2], 0.0)


def test_safe_hstack_sparse_dense():
    a = csr_matrix(np.ones((2, 2)))
    b = np.zeros((2, 1))
    result = utils.safe_hstack_hybrid(a, b)
    assert issparse(result)
    assert result.shape == (2, 3)


def test_safe_hstack_dense_sparse():
    a = np.ones((2, 2))
    b = csr_matrix(np.zeros((2, 1)))
    result = utils.safe_hstack_hybrid(a, b)
    assert issparse(result)
    assert result.shape == (2, 3)


def test_safe_hstack_sparse_sparse():
    a = csr_matrix(np.ones((2, 2)))
    b = csr_matrix(np.zeros((2, 1)))
    result = utils.safe_hstack_hybrid(a, b)
    assert issparse(result)
    assert result.shape == (2, 3)


def test_safe_hstack_force_dense_sparse_sparse():
    a = csr_matrix(np.ones((2, 2)))
    b = csr_matrix(np.zeros((2, 1)))
    result = utils.safe_hstack_hybrid(a, b, force_dense=True)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3)


def test_safe_hstack_1d_array():
    a = np.ones((2, 2))
    b = np.array([0, 1])
    result = utils.safe_hstack_hybrid(a, b)
    assert result.shape == (2, 3)
    assert np.allclose(result[:, 2], [0, 1])


# -------------------------------------------------------------------
# DEFAULT_REGRESSORS tests
# -------------------------------------------------------------------

def test_default_regressors_core():
    core_keys = ["linear", "ridge", "lasso", "elastic_net", "random_forest", "gradient_boosting", "svr", "knn"]
    for key in core_keys:
        assert key in DEFAULT_REGRESSORS
        assert hasattr(DEFAULT_REGRESSORS[key], "fit")


def test_default_regressors_optional(monkeypatch):
    # Dummy optional regressors
    class DummyXGB: 
        def __init__(self, **kwargs): self.kwargs = kwargs
    class DummyLGBM:
        def __init__(self, **kwargs): self.kwargs = kwargs
    class DummyCatBoost:
        def __init__(self, **kwargs): self.kwargs = kwargs

    monkeypatch.setattr(utils, "XGBRegressor", DummyXGB)
    monkeypatch.setattr(utils, "LGBMRegressor", DummyLGBM)
    monkeypatch.setattr(utils, "CatBoostRegressor", DummyCatBoost)

    # Rebuild defaults manually
    defaults = DEFAULT_REGRESSORS.copy()
    if utils.XGBRegressor is not None:
        defaults["xgboost"] = utils.XGBRegressor(random_state=42)
    if utils.LGBMRegressor is not None:
        defaults["lightgbm"] = utils.LGBMRegressor(random_state=42)
    if utils.CatBoostRegressor is not None:
        defaults["catboost"] = utils.CatBoostRegressor(random_state=42, verbose=0)

    assert "xgboost" in defaults
    assert "lightgbm" in defaults
    assert "catboost" in defaults
    assert isinstance(defaults["xgboost"], DummyXGB)
    assert isinstance(defaults["lightgbm"], DummyLGBM)
    assert isinstance(defaults["catboost"], DummyCatBoost)


# -------------------------------------------------------------------
# get_default_regressors fallback (optional regressors not installed)
# -------------------------------------------------------------------

def test_get_default_regressors_xgboost_none(monkeypatch):
    monkeypatch.setattr(utils, "XGBRegressor", None)
    defaults = utils.get_default_regressors()
    # xgboost should NOT be in defaults
    assert "xgboost" not in defaults

def test_get_default_regressors_lightgbm_none(monkeypatch):
    monkeypatch.setattr(utils, "LGBMRegressor", None)
    defaults = utils.get_default_regressors()
    # lightgbm should NOT be in defaults
    assert "lightgbm" not in defaults

def test_get_default_regressors_catboost_none(monkeypatch):
    monkeypatch.setattr(utils, "CatBoostRegressor", None)
    defaults = utils.get_default_regressors()
    # catboost should NOT be in defaults
    assert "catboost" not in defaults

