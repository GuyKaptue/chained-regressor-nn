# tests/test_pipeline.py
import numpy as np # type:ignore
import pandas as pd #type:ignore
import pytest # type:ignore
from scipy.sparse import csr_matrix # type:ignore
from chainedregressornn import ChainedRegressorNN, NeuralRegressor, safe_hstack_hybrid # type:ignore


def test_residual_stacking(generate_sample_orders):
    df = generate_sample_orders.copy()
    model = ChainedRegressorNN(target="profit", use_nn=False, auto_rank=False)
    model.fit(df, verbose=False)

    # manual reconstruction of predict()
    df_proc = df.copy()
    df_proc = model._preprocess_dates(df_proc)
    df_proc = model._clean_columns(df_proc)
    X_current = model.preprocessor.transform(df_proc)

    manual_preds = None
    for name, stage in model.pipeline:
        if isinstance(stage, NeuralRegressor):
            preds = stage(model._to_tensor(X_current)).detach().numpy()
            preds_sparse = csr_matrix(preds.reshape(-1, 1))
            X_current = safe_hstack_hybrid(X_current, preds_sparse)
        else:
            preds = stage.predict(X_current)
            X_current = safe_hstack_hybrid(X_current, preds.reshape(-1, 1))
        manual_preds = preds

    # pipeline predictions
    final_preds = model.predict(df)

    # they should match exactly
    assert np.allclose(final_preds, manual_preds, atol=1e-6)

@pytest.mark.parametrize("use_nn,hidden_dims", [
    (False, []),
    (True, [16]),
    (True, [32, 16])
])
def test_pipeline_variants(generate_sample_orders, use_nn, hidden_dims):
    df = generate_sample_orders.copy()
    y = df["profit"]
    X = pd.get_dummies(df.drop(columns=["profit"]), drop_first=True)

    model = ChainedRegressorNN(target="profit", use_nn=use_nn, hidden_dims=hidden_dims)
    model.fit(X, y, verbose=False, nn_epochs=3)

    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.reshape(-1).shape == (len(X),)
