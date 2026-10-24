# Data Learning

This repository documents my journey of learning **Data Science, Machine Learning, and related programming skills** through practical projects.

The goal is to learn concepts by applying them to real-world datasets, while gradually improving the projects as I learn new tools and techniques.

## Learning Path

My current learning path is:

```text
Python
   ↓
NumPy / Pandas
   ↓
Data Cleaning
   ↓
Data Visualization
   ↓
Data Analysis
   ↓
Statistics
   ↓
Machine Learning
   ↓
Model Evaluation
```

The exact order may change as I learn and explore different topics.

## Projects

### Global House Price Analysis

A data analysis project using residential property price data from the **Bank for International Settlements (BIS)**.

This project currently includes:

- Cleaning and visualizing quarterly residential property price indices
- A one-quarter-ahead forecasting experiment for Korea's nominal price index, comparing a persistence baseline with scikit-learn regressors

See the [house price project](./house_price/README.md), the [cleaning and visualization notebook](./house_price/house_pricenotebooks/hp_cleaning_visualization.ipynb), and the [forecasting notebook](./house_price/house_pricenotebooks/02_machine_learning_modeling.ipynb).

### Machine Learning Models: Inside Out

An interactive guide to how machine learning models work, with diagrams and a separate Python file of runnable examples.

[Open the interactive guide](https://mashpotatoio.github.io/dat_learning/machine_learning/ML_Models_Inside_Out_EN.html) · [View the HTML source](./machine_learning/ML_Models_Inside_Out_EN.html) · [View the Python examples](./machine_learning/ml_models_examples.py)

### Machine Learning Models: Inside Out

An interactive guide to how machine learning models work, with diagrams and runnable examples.

[Open the interactive guide](https://mashpotatoio.github.io/dat_learning/machine_learning/ML_Models_Inside_Out_EN.html) · [View the HTML source](./machine_learning/ML_Models_Inside_Out_EN.html)

## Repository Structure

```text
dat_learning/
├── README.md
├── .gitignore
├── house_price/
│   ├── README.md
│   ├── house_pricedata/
│   │   └── nominal-index.csv
│   └── house_pricenotebooks/
│       ├── hp_cleaning_visualization.ipynb
│       └── 02_machine_learning_modeling.ipynb
└── machine_learning/
    ├── ML_Models_Inside_Out_EN.html
    └── ml_models_examples.py
```

## Running the examples

Install Python, Jupyter, NumPy, Pandas, Matplotlib, Seaborn, and scikit-learn. Open the house price notebooks in Jupyter with the working directory set to `house_price/house_pricenotebooks/` so their relative path to `house_pricedata/nominal-index.csv` resolves. Run the standalone examples with `python machine_learning/ml_models_examples.py` from the repository root. The HTML guide can be opened in a browser.

## Tools

The main tools I am currently learning and using include:

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- Jupyter Notebook
- Git / GitHub

More tools will be added as the projects develop.

## Purpose

This repository is primarily a **learning log and project portfolio**.

Instead of only following tutorials, I want to use real datasets, investigate questions, make mistakes, and improve my work over time.

Each project may start with basic data analysis and gradually develop into more advanced statistical or machine learning work.
