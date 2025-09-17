"""
core.py
=======

Main implementation of the ChainedRegressorNN pipeline.

This module defines the ChainedRegressorNN class, which chains together
multiple regression models (from scikit-learn and optionally neural networks)
in sequence. Each stage's predictions are appended to the feature set for
subsequent stages, allowing for stacked learning with both traditional ML
models and deep learning components.

Author
------
Guy Kaptue
"""

import numpy as np  # type:ignore
import pandas as pd # type:ignore
import torch # type:ignore
from collections import OrderedDict # type:ignore
from sklearn.base import clone # type:ignore
from sklearn.compose import ColumnTransformer # type:ignore
from sklearn.preprocessing import StandardScaler, OneHotEncoder # type:ignore
from sklearn.model_selection import cross_val_score, KFold # type:ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # type:ignore
from scipy.sparse import issparse, csr_matrix # type:ignore

from .neural import NeuralRegressor, train_neural_network
from .utils import safe_hstack_hybrid, DEFAULT_REGRESSORS

import warnings
from sklearn.exceptions import ConvergenceWarning # type: ignore

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names"
)

class ChainedRegressorNN:
    """
    A chained regression pipeline combining traditional ML models with optional neural networks.

    Parameters
    ----------
    target : str
        Name of the target column in the input DataFrame.
    regressors : dict, optional
        Dictionary of regressor name → estimator objects. Defaults to DEFAULT_REGRESSORS.
    use_nn : bool, default=True
        Whether to insert neural network stages between regressors.
    drop_ids : bool, default=True
        Whether to drop columns containing 'id' or 'name' (case-insensitive).
    auto_rank : bool, default=True
        Whether to rank regressors by cross-validated RMSE before training.
    cv_folds : int, default=5
        Number of folds for cross-validation when ranking regressors.
    hidden_dims : list[int], optional
        Hidden layer sizes for neural network stages.

    Attributes
    ----------
    pipeline : list
        Sequence of (name, model) tuples representing the trained pipeline.
    preprocessor : ColumnTransformer
        Preprocessing transformer for numeric and categorical features.
    feature_cols : list[str]
        List of feature column names after preprocessing.
    """

    def __init__(self, target, regressors=None, use_nn=True, drop_ids=True,
                 auto_rank=True, cv_folds=5,
                 hidden_dims=None):
        if hidden_dims is None:
            hidden_dims = [512, 256, 256, 128, 64, 64, 32, 16]

        self.regressors = regressors if regressors is not None else DEFAULT_REGRESSORS.copy()
        self.target = target
        self.use_nn = use_nn
        self.drop_ids = drop_ids
        self.auto_rank = auto_rank
        self.cv_folds = cv_folds
        self.hidden_dims = hidden_dims
        self.pipeline = []
        self.preprocessor = None
        self.feature_cols = None

    def _preprocess_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Expand datetime columns into year, month, and day components."""
        for col in df.select_dtypes(include=["datetime64[ns]"]).columns:
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
        return df

    def _clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop ID/name columns if drop_ids is True."""
        if self.drop_ids:
            drop_cols = [c for c in df.columns if "id" in c.lower() or "name" in c.lower()]
            df = df.drop(columns=drop_cols, errors="ignore")
        return df

    def _build_preprocessor(self, df: pd.DataFrame):
        """Build a ColumnTransformer for numeric and categorical preprocessing."""
        numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if self.target in numeric_features:
            numeric_features.remove(self.target)
        if self.target in categorical_features:
            categorical_features.remove(self.target)
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ]
        )
        return preprocessor, numeric_features + categorical_features

    def _rank_regressors_by_rmse(self, df: pd.DataFrame) -> OrderedDict:
        """Rank regressors by cross-validated RMSE."""
        y = df[self.target].values
        X = df.drop(columns=[self.target])
        preprocessor, _ = self._build_preprocessor(df)
        X_proc = preprocessor.fit_transform(X)
        results = []
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        for name, model in self.regressors.items():
            try:
                scores = cross_val_score(model, X_proc, y,
                                         scoring="neg_root_mean_squared_error", cv=cv)
                avg_rmse = -np.mean(scores)
                results.append((name, avg_rmse))
            except Exception as e:
                print(f" Skipping {name} due to error: {e}")
        results.sort(key=lambda x: x[1])
        print("\n Model ranking by CV RMSE:")
        for name, rmse in results:
            print(f"   {name}: {rmse:.4f}")
        return OrderedDict((name, self.regressors[name]) for name, _ in results)

    def fit(self, X, y=None, verbose=False, nn_epochs=500):
        """
        Fit the chained regression pipeline.

        Parameters
        ----------
        X : pandas.DataFrame
            Training data. Can include the target column if `y` is None.
        y : array-like, optional
            Target values. If None, `X` must contain the target column.
        verbose : bool, default=False
            Whether to print training progress.
        nn_epochs : int, default=500
            Number of epochs for neural network training.
        """
        # Accept both (df_with_target) and (X, y) styles
        if y is None:
            if self.target not in X.columns:
                raise ValueError(f"Target column '{self.target}' not found in DataFrame.")
            y = X[self.target].values
            X = X.drop(columns=[self.target])
        else:
            y = np.asarray(y)

        # Defensive copy
        df = X.copy()

        # Preprocess dates and drop ID/name columns
        df = self._preprocess_dates(df)
        df = self._clean_columns(df)

        # Reattach target for ranking if needed
        df_with_target = df.copy()
        df_with_target[self.target] = y

        # Optionally rank regressors
        if self.auto_rank:
            self.regressors = self._rank_regressors_by_rmse(df_with_target)

        # Build preprocessor and transform features
        preprocessor, feature_cols = self._build_preprocessor(df_with_target)
        self.preprocessor = preprocessor
        self.feature_cols = feature_cols

        X_proc = preprocessor.fit_transform(df_with_target)
        X_current = X_proc
        self.pipeline = []

        # Train each regressor (and optional NN stage)
        for i, (name, reg) in enumerate(self.regressors.items()):
            if verbose:
                print(f"\nTraining regressor {i+1}/{len(self.regressors)}: {name}")

            reg_clone = clone(reg)
            reg_clone.fit(X_current, y)
            preds = reg_clone.predict(X_current)
            X_current = safe_hstack_hybrid(X_current, preds)
            self.pipeline.append((name, reg_clone))

            if self.use_nn and i < len(self.regressors) - 1:
                if verbose:
                    print(f"  Training neural net after {name} (stage {i+1})...")
                X_nn = safe_hstack_hybrid(X_current, np.zeros((X_current.shape[0], 0)), force_dense=True)
                preds_nn, nn_model = train_neural_network(
                    X_nn, y,
                    input_dim=X_nn.shape[1],
                    hidden_dims=self.hidden_dims,
                    epochs=nn_epochs,
                    verbose=verbose
                )
                preds_nn_sparse = csr_matrix(preds_nn)
                X_current = safe_hstack_hybrid(X_current, preds_nn_sparse)
                self.pipeline.append((f"nn_{i+1}", nn_model))

        if verbose:
            print("\nTraining complete! Pipeline built with:")
            for step_name, _ in self.pipeline:
                print(f"  • {step_name}")

    def _to_tensor(self, X):
        """Ensure input is a float32 torch.Tensor, handling sparse/dense/DF."""
        if issparse(X):
            X = X.toarray()
        elif hasattr(X, "values"):
            X = X.values
        return torch.as_tensor(X, dtype=torch.float32)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for new data.

        Parameters
        ----------
        df : pandas.DataFrame
            Input features.

        Returns
        -------
        numpy.ndarray
            Predicted values.
        """
        df = df.copy()
        df = self._preprocess_dates(df)
        df = self._clean_columns(df)
        X_current = self.preprocessor.transform(df)

        for name, model in self.pipeline:
            if isinstance(model, NeuralRegressor):
                preds = model(self._to_tensor(X_current)).detach().numpy()
                preds_sparse = csr_matrix(preds)
                X_current = safe_hstack_hybrid(X_current, preds_sparse)
            else:
                preds = model.predict(X_current)
                X_current = safe_hstack_hybrid(X_current, preds)

        return np.asarray(X_current[:, -1]).ravel()

    def evaluate(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Evaluate the trained pipeline on a given dataset.

        This method computes common regression metrics (RMSE, MAE, R²)
        using the pipeline's predictions on the provided DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            Evaluation dataset containing the same features and target column
            used during training.

        Returns
        -------
        dict of str to float
            Dictionary with keys:
            - "rmse": Root Mean Squared Error
            - "mae": Mean Absolute Error
            - "r2": Coefficient of determination (R² score)

        Notes
        -----
        - The method internally calls `predict`, so it is safe for pipelines
          containing both scikit-learn regressors and NeuralRegressor stages.
        - Input DataFrame will be preprocessed in the same way as during training.
        """
        df = df.copy()
        df = self._preprocess_dates(df)
        df = self._clean_columns(df)

        y_true = df[self.target].values
        y_pred = self.predict(df)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(" Evaluation Results:")
        print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

        return {"rmse": rmse, "mae": mae, "r2": r2}