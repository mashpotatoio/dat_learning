# Global House Price Analysis

A data analysis project exploring residential property prices across different countries using data from the **Bank for International Settlements (BIS)**.

This is the first major project in my data science learning journey. I am using it to practice working with real-world data and gradually develop skills in **data cleaning, visualization, statistical analysis, and machine learning**.

## Dataset

The included `house_pricedata/nominal-index.csv` contains quarterly nominal residential property price indices for multiple countries, with 2010 = 100. Its columns are `date`, `country`, and `price`; an index value is not a home price in currency.

## Data Source

**Bank for International Settlements (BIS)**  
Residential Property Price Statistics

https://www.bis.org/statistics/pp.htm

The dataset used in this project is the BIS selected series dataset.

## Project Structure

```text
house_price/
│
├── README.md
│
├── house_pricedata/
│   └── nominal-index.csv
└── house_pricenotebooks/
    ├── hp_cleaning_visualization.ipynb
    └── 02_machine_learning_modeling.ipynb
```

## Phase 1 — Data Cleaning & Visualization

The first phase focuses on understanding and preparing the dataset.

Topics currently being practiced:

- Loading CSV data with Pandas
- Inspecting DataFrames
- Handling missing values
- Working with dates
- Extracting years from dates
- Filtering data
- Counting observations
- Grouping data
- Calculating basic statistics
- Creating visualizations with Matplotlib

## Phase 2 — Machine Learning Modeling

The [forecasting notebook](./house_pricenotebooks/02_machine_learning_modeling.ipynb) uses Korea's quarterly nominal index to predict the following quarter. It compares a persistence baseline with regression models, uses chronological training and holdout periods, and evaluates errors in index points. This is a historical learning exercise, not a current market forecast.

Open the notebooks in Jupyter from `house_price/house_pricenotebooks/` so the relative path to the included CSV resolves. The modeling notebook also requires scikit-learn.

## Future Development

As I learn more data science concepts, this project may be extended into:

```text
Data Cleaning
      ↓
Visualization
      ↓
Data Analysis
      ↓
Statistics
      ↓
Machine Learning
```

Future work may include statistical analysis and experimenting with machine learning models where appropriate.

## Tools

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- Jupyter Notebook
- Git / GitHub

## Goal

The main goal of this project is to learn how to work with a real-world dataset from start to finish.

Rather than treating the project as a finished analysis, I will gradually improve it as I learn new concepts and techniques.

The notebook files therefore represent the progression of my learning and experimentation.
