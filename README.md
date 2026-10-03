# Global House Price Analysis

A data analysis project exploring residential property prices across different countries using data from the **Bank for International Settlements (BIS)**.

This is the first major project in my data science learning journey. I am using it to practice working with real-world data and gradually develop skills in **data cleaning, visualization, statistical analysis, and machine learning**.

## Dataset

The dataset contains residential property price statistics from different countries.

It provides four types of indicators:

- **Nominal Index** — house price index, 2010 = 100
- **Nominal Year** — nominal year-on-year changes, in percent
- **Real Index** — real house price index, 2010 = 100
- **Real Year** — real year-on-year changes, in percent

The dataset contains quarterly observations for multiple countries.

Real house price series are nominal house price series adjusted using the Consumer Price Index (CPI).

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
├── data/
│   ├── nominal_index.csv
│   ├── nominal_year.csv
│   ├── real_index.csv
│   └── real_year.csv
│
└── notebooks/
    └── 01_cleaning_visualization.ipynb
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

## Phase 2 — Data Analysis

The next stage will focus on asking questions about the data rather than only visualizing it.

Possible areas of analysis include:

- Changes in house prices over time
- Yearly house price trends
- Comparing trends between countries
- Nominal vs. real house prices
- Identifying periods of large price changes
- Exploring differences in housing markets between countries

The specific questions will be refined as the analysis develops.

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
- Jupyter Notebook
- Git / GitHub

## Goal

The main goal of this project is to learn how to work with a real-world dataset from start to finish.

Rather than treating the project as a finished analysis, I will gradually improve it as I learn new concepts and techniques.

The notebook files therefore represent the progression of my learning and experimentation.