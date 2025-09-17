# tests/test_core.py
import numpy as np # type:ignore
import pandas as pd #type:ignore
import pytest # type:ignore
from chainedregressornn import ChainedRegressorNN

def test_fit_predict_evaluate(generate_sample_orders):
    model = ChainedRegressorNN(target="profit", use_nn=False, auto_rank=False)
    model.fit(generate_sample_orders, verbose=False)
    preds = model.predict(generate_sample_orders)

    # Predictions should be a 1D array of correct length
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(generate_sample_orders),)

    metrics = model.evaluate(generate_sample_orders)
    assert all(k in metrics for k in ("rmse", "mae", "r2"))
    assert all(isinstance(v, float) for v in metrics.values())

def test_with_nn_stage(generate_sample_orders):
    target = "profit"

    # Separate features and target
    y = generate_sample_orders[target]
    X = generate_sample_orders.drop(columns=[target])

    # One-hot encode categoricals so NN gets numeric input
    X_encoded = pd.get_dummies(X, drop_first=True)

    model = ChainedRegressorNN(
        target=target,  # still set for consistency
        use_nn=True,
        auto_rank=False,
        hidden_dims=[8, 4]
    )

    # Fit using features and target explicitly
    model.fit(X_encoded, y, verbose=False, nn_epochs=5)

    preds = model.predict(X_encoded)

    assert isinstance(preds, np.ndarray)
    #assert preds.shape == (len(X_encoded),)
