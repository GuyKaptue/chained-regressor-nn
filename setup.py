from setuptools import setup, find_packages # type: ignore
import pathlib

# Base directory
here = pathlib.Path(__file__).parent.resolve()

# Read README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# Read requirements, ignoring comments and empty lines
def parse_requirements(filename):
    with open(here / filename, encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

requirements = parse_requirements("requirements.txt")

setup(
    name="chained-regressor-nn",
    version="0.1.1",
    author="Guy Kaptue",
    author_email="guykaptue24@gmail.com",
    description="A chained regression approach combining traditional ML models with neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GuyKaptue/chained-regressor-nn",
    project_urls={
        "Bug Tracker": "https://github.com/GuyKaptue/chained-regressor-nn/issues",
        "Documentation": "https://github.com/GuyKaptue/chained-regressor-nn#readme",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(exclude=["tests*", "docs*"]),
    python_requires=">=3.8, <4",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "optional": [
            "xgboost>=2.0.0",
            "lightgbm>=4.0.0",
            "catboost>=1.2.0",
        ],
        "full": [
            "xgboost>=2.0.0",
            "lightgbm>=4.0.0",
            "catboost>=1.2.0",
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    data_files=[("docs", ["docs/kap_formula_paper.pdf"])],
    include_package_data=True,
    zip_safe=False,
)
