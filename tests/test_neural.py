# tests/test_neural.py
import numpy as np # type:ignore
import torch # type:ignore

from chainedregressornn import NeuralRegressor, train_neural_network, set_seed

def test_neural_regressor_forward():
    set_seed(42)
    model = NeuralRegressor(input_dim=4, hidden_dims=[8, 4], dropout=0.1)
    x = torch.randn(5, 4)
    out = model(x)
    assert out.shape == (5, 1)

def test_train_neural_network():
    set_seed(42)
    X = np.random.randn(20, 4)
    y = np.random.randn(20)
    preds, model = train_neural_network(X, y, input_dim=4, hidden_dims=[8, 4], epochs=5, verbose=False)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (20, 1)
    assert isinstance(model, NeuralRegressor)
