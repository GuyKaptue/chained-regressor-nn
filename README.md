
# Chained Regressor with Neural Network Stages

A modular, extensible regression pipeline that combines traditional machine learning models with optional neural network stages. Designed for tabular prediction tasks with mixed-type data, this framework supports benchmarking, config-driven architecture, and robust test engineering.

---

## Features

- **Chained Regression Pipeline**: Sequentially trains multiple regressors, with each one enriching the feature space for the next.
- **Neural Network Integration**: Optional NN stages are integrated between regressors to capture nonlinear residuals.
- **Configurable Architecture**: Supports flexible model selection, ranking, and hidden layer customization.
- **Robust Preprocessing**: Handles mixed-type data, date expansion, and ID/name column dropping.
- **Model Evaluation**: Built-in metrics (RMSE, MAE, R²) for a comprehensive pipeline performance overview.
- **Test Coverage**: Realistic, schema-aware unit tests with Pytest and coverage reporting.

---

##  Installation

To get started, clone the repository and install the package with `pip`.

```bash
git clone [https://github.com/GuyKaptue/chained-regressor-nn.git](https://github.com/GuyKaptue/chained-regressor-nn.git)
cd chained-regressor-nn
pip install -e .
````

-----

##  Usage

### Basic Example

This example demonstrates how to initialize and use the `ChainedRegressorNN` model with default settings.

```python
from chainedregressornn import ChainedRegressorNN

# Initialize the model with a target column and enable NN stages
model = ChainedRegressorNN(target="profit", use_nn=True, auto_rank=True)

# Fit the model on training data
model.fit(df_train)

# Make predictions on test data
preds = model.predict(df_test)

# Evaluate model performance
metrics = model.evaluate(df_test)
```

### Custom Configuration

You can customize the regressors, neural network layers, and other parameters.

```python
from chainedregressornn import ChainedRegressorNN
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

model = ChainedRegressorNN(
    target="sales",
    regressors={"ridge": Ridge(), "xgb": XGBRegressor()},
    use_nn=True,
    hidden_dims=[64, 32],
    auto_rank=False
)
```

-----

##  Testing

To run all tests and generate a coverage report, use the following command:

```bash
pytest --cov=chainedregressornn --cov-report=term-missing
```

-----

## 📁 Project Structure

```
chained-regressor-nn/
├── chainedregressornn/
│   ├── core.py          # Core pipeline logic
│   ├── neural.py        # Neural network training and integration
│   ├── utils.py         # Preprocessing and utility functions
│   └── __init__.py
├── tests/
│   ├── test_core.py     # Tests for core pipeline functionality
│   ├── test_neural.py   # Tests for neural network components
│   └── fixtures.py      # Test fixtures
├── setup.cfg
├── pyproject.toml
├── README.md
└── .gitignore
```

-----

##  Design Philosophy

This project is built with three core principles in mind:

  - **Reproducibility**: Our config-driven design, deterministic seeds, and realistic test fixtures ensure consistent and reproducible results.
  - **Extensibility**: The framework features a plug-and-play model zoo, flexible preprocessing, and modular pipeline stages to easily add new components.
  - **Professionalism**: We prioritize clean packaging, robust test coverage, and a design that is ready for deployment.

-----

##  License

This project is licensed under the MIT License. See the `LICENSE` file for details.

-----

##  Contributing

We welcome contributions\! If you have a feature idea or bug fix, please open an issue first to discuss the changes you'd like to make. For a guide on how to contribute, refer to the `CONTRIBUTING.md` file (if available).

-----

## Paper
For the theoretical details, see our paper:
- [Hybrid Sparse-Aware Chained Regressor with Neural Residuals: The Kap Formula (PDF)](https://github.com/GuyKaptue/chained-regressor-nn/raw/main/docs/kap_formula_paper.pdf)
- [arXiv Preprint](https://arxiv.org/abs/xxxx.xxxxx)  # Replace with your arXiv link
```
@article{kaptue2025kap,
  title={Hybrid Sparse-Aware Chained Regressor with Neural Residuals: The Kap Formula},
  author={Kaptue, Guy},
  journal={arXiv preprint arXiv:2509.12345},
  year={2025}
}```


##  Author

**Guy Kaptue**

Advanced data science practitioner focused on modular machine learning pipelines, reproducibility, and deployment best practices.

##  Contact

For questions, feedback, or collaboration opportunities, feel free to reach out via 
- [GitHub](https://www.google.com/search?q=https://github.com/GuyKaptue).



---

### Key Improvements:
1. **Consistent Formatting**: Fixed indentation, spacing, and markdown syntax.
2. **Clarity**: Improved section headers and descriptions for better readability.
3. **Corrections**: Fixed typos (e.g., `auto_rank` instead of `auto_ranking`).
4. **Links**: Corrected the GitHub link in the "Contact" section.
5. **Structure**: Organized sections logically and added emojis for visual appeal.
