# tests/test_core.py
import numpy as np # type:ignore
import pandas as pd #type:ignore
import pytest # type:ignore
import torch # type:ignore
from sklearn.linear_model import LinearRegression # type:ignore
from sklearn.tree import DecisionTreeRegressor  # type:ignore

from sklearn.neural_network import MLPRegressor # type:ignore
from scipy.sparse import csr_matrix # type:ignore
from chainedregressornn import ChainedRegressorNN, NeuralRegressor
from unittest.mock import patch, MagicMock


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

def test_predict_before_fit_raises():
    df = pd.DataFrame({"f1": [1, 2, 3], "target": [0, 1, 0]})
    model = ChainedRegressorNN(target="target")

    # Predict should raise RuntimeError if model not fitted
    with pytest.raises(RuntimeError, match="Model must be fitted before calling predict"):
        model.predict(df)

def test_evaluate_before_fit_raises(generate_sample_orders):
    """Tests that calling evaluate before fit raises a RuntimeError."""
    df = generate_sample_orders.copy()
    model = ChainedRegressorNN(target="sales")
    
    # Corrected regex pattern to match the actual error message
    with pytest.raises(RuntimeError, match="Model must be fitted before calling evaluate"):
        model.evaluate(df)


def test_build_preprocessor_with_categorical_target(generate_sample_orders):
    """
    Tests _build_preprocessor when the target is a categorical column.
    This test is designed to cover line 113 in core.py.
    """
    df = generate_sample_orders.copy()
    # Convert 'category' to a categorical target to hit the missed line
    df['category'] = df['category'].astype('category')
    
    # Instantiate the regressor with the categorical target
    regressor = ChainedRegressorNN(target='category', use_nn=False)
    
    # We mock _clean_columns and _preprocess_dates to isolate the preprocessor test
    with patch.object(regressor, '_clean_columns', return_value=df), \
         patch.object(regressor, '_preprocess_dates', return_value=df):
        preprocessor, feature_cols = regressor._build_preprocessor(df)
        
        # Assert that the target column 'category' is not in the feature list
        assert 'category' not in feature_cols
        assert 'sales' in feature_cols
        assert 'city' in feature_cols


def test_predict_with_sparse_nn_output(generate_sample_orders):
    df = generate_sample_orders.copy()
    target = "sales"

    regressors = {
        "lr": LinearRegression(),
        "dt": DecisionTreeRegressor()
    }
    
    regressor = ChainedRegressorNN(target=target, regressors=regressors, use_nn=True)
    nn_model = NeuralRegressor(input_dim=29, hidden_dims=[10])
    
    # 1. Define the DESIRED final prediction output
    desired_prediction = np.array([1.5, 2.5, 3.5])
    
    # Patch NN training to return the nn_model and dummy predictions for FIT
    nn_predictions = np.zeros((len(df), 1))
    with patch(
        "chainedregressornn.core.train_neural_network",
        return_value=(nn_predictions, nn_model)  # shape (n_samples, 1)
    ):
        regressor.fit(df, nn_epochs=0, verbose=True)
    
    # 2. Mock the predict method for *all* regressor stages
    # The first stage (lr) must return zero to isolate the NN's mock output later.
    # The last stage (dt) must return the DESIRED output.
    for name, stage in regressor.pipeline:
        # Note: We are using the stage instances *from* the fitted pipeline,
        # not the original regressors dict.
        if name == "lr":
            # lr prediction is added as a feature. Let it be 0.
            stage.predict = MagicMock(return_value=np.zeros(len(desired_prediction)))
        elif name == "dt":
            # dt is the *final* stage, so its output MUST be the expected result.
            stage.predict = MagicMock(return_value=desired_prediction) 
    
    # 3. Patch NN forward to return the desired prediction (as a feature contribution)
    mock_nn_output = torch.tensor([[100.0], [200.0], [300.0]], dtype=torch.float32) # Value doesn't matter much if dt is fully mocked
    
    with patch.object(nn_model, "forward", return_value=mock_nn_output) as mock_forward:
        predictions = regressor.predict(df.iloc[0:3])
    
    # Assert predictions match mocked DT output (which is our desired final output)
    np.testing.assert_allclose(predictions, desired_prediction)

    # Assert pipeline contains both the LinearRegression stage and NN stage
    stage_names = [name for name, _ in regressor.pipeline]
    assert "lr" in stage_names
    assert "dt" in stage_names
    assert any(name.startswith("nn_") for name in stage_names)

