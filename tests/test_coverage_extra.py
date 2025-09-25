# tests/test_coverage_extra.py
import pytest # type:ignore
import numpy as np # type:ignore
import pandas as pd # type:ignore
from scipy.sparse import csr_matrix # type:ignore
import torch # type:ignore

from chainedregressornn.core import ChainedRegressorNN
from chainedregressornn.neural import NeuralRegressor, train_neural_network
from chainedregressornn.utils import safe_hstack_hybrid

from sklearn.base import BaseEstimator, RegressorMixin # type:ignore


# --------------------------------------------------------
# Test NeuralRegressor residual forward pass
# --------------------------------------------------------
def test_neural_regressor_residual_forward():
    X = np.random.rand(5, 3)
    nn = NeuralRegressor(input_dim=3, hidden_dims=[4, 4], residual=True)
    y = nn(X).detach().numpy()
    assert y.shape == (5, 1)

# --------------------------------------------------------
# Test train_neural_network with verbose=False
# --------------------------------------------------------
def test_train_neural_network_verbose_false():
    X = np.random.rand(5, 2)
    y = np.random.rand(5)
    preds, model = train_neural_network(X, y, input_dim=2, hidden_dims=[4, 4], epochs=5, verbose=False)
    assert preds.shape == (5, 1)
    assert isinstance(model, NeuralRegressor)

# --------------------------------------------------------
# Test fit raises error if target missing
# --------------------------------------------------------
def test_fit_missing_target_raises():
    df = pd.DataFrame(np.random.rand(5, 2), columns=["f1", "f2"])
    cr = ChainedRegressorNN(target="y", use_nn=False)
    with pytest.raises(ValueError):
        cr.fit(df)

# --------------------------------------------------------
# Test predict works with only NeuralRegressor and sparse input
# --------------------------------------------------------
def test_predict_only_nn_sparse():
    import torch  # type:ignore
    import pandas as pd  # type:ignore
    import numpy as np  # type:ignore

    # --- synthetic data ---
    df = pd.DataFrame(np.random.rand(5, 2), columns=["f1", "f2"])
    y = np.random.rand(5)

    # --- chained regressor with NN only ---
    cr = ChainedRegressorNN(target="y", use_nn=True, auto_rank=False, regressors={})

    # Minimal NeuralRegressor: 2 input features -> 1 output
    nn_model = NeuralRegressor(input_dim=2, hidden_dims=[4, 4])
    cr.pipeline = [("nn_0", nn_model)]

    # --- dummy preprocessor ---
    class DummyPreprocessor:
        def transform(self, df_input):
            # explicitly keep only the feature columns
            return df_input[["f1", "f2"]].values

    cr.preprocessor = DummyPreprocessor()
    cr._preprocess_dates = lambda df: df
    cr._clean_columns = lambda df: df
    cr.feature_cols = ["f1", "f2"]

    # add target column
    df["y"] = y

    # patch tensor conversion
    cr._to_tensor = lambda X: torch.tensor(X, dtype=torch.float32)

    # --- run prediction ---
    preds = cr.predict(df)

    # ensure flat shape for checking
    preds = preds.ravel()

    # --- assertions ---
    assert preds.shape == (5,)
    assert np.all(np.isfinite(preds))




# --------------------------------------------------------
# Test _rank_regressors_by_rmse handles regressor exceptions
# --------------------------------------------------------
def test_rank_regressors_exception():
    df = pd.DataFrame(np.random.rand(5, 2), columns=["f1", "f2"])
    df["y"] = np.random.rand(5)
    cr = ChainedRegressorNN(target="y", auto_rank=True)

    # Regressor that fails during fit but is sklearn-compatible
    class BadReg(BaseEstimator, RegressorMixin):
        def fit(self, X, y):
            raise RuntimeError("fail")
        def predict(self, X):
            return np.zeros(X.shape[0])

    cr.regressors = {"bad": BadReg()}
    ranked = cr._rank_regressors_by_rmse(df)
    assert "bad" not in ranked  # Should be skipped but not crash

# --------------------------------------------------------
# Test safe_hstack with sparse/dense mixed arrays (utility)
# --------------------------------------------------------
def test_safe_hstack_sparse_dense_mix():
    from chainedregressornn.utils import safe_hstack_hybrid
    import numpy as np # type:ignore
    from scipy.sparse import csr_matrix # type:ignore

    a = csr_matrix(np.random.rand(5, 2))
    b = np.random.rand(5, 1)
    stacked = safe_hstack_hybrid(a, b)
    assert stacked.shape == (5, 3)