# Flight Ticket Price Prediction

This project aims to predict flight ticket prices using Gradient Boosting Regressor models and compare the performance between two data handling and modeling pipelines:

- Traditional method using Pandas and Scikit-learn
- Distributed processing using PySpark MLlib

---

## Dataset

The dataset used is available on Kaggle:  
[Flight Fare Prediction Dataset](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)

It contains features such as:
- Airline
- Source and destination cities
- Departure and arrival times
- Flight class
- Number of stops
- Days left before departure
- Target variable: `price`

---

## Project Structure

```plaintext
├── EDA.ipynb                   # Exploratory Data Analysis with visual insights
├── Normal_GradientBoosting.ipynb   # Pandas + Scikit-learn pipeline
├── PySpark_GradientBoosting.ipynb  # PySpark MLlib pipeline
├── README.md
```

---

## Objective

To compare model performance and data handling differences between the traditional and PySpark-based ML pipelines.

---

## Methodology Overview

### Exploratory Data Analysis (`EDA.ipynb`)

- Univariate and multivariate visualizations
- Correlation heatmap
- Insights into how each feature affects flight ticket prices

### Preprocessing

- **Normal method**: Preprocessed using Pandas including label encoding and basic cleanup
- **PySpark method**: Preprocessed using `StringIndexer`, `OneHotEncoder`, and `VectorAssembler` inside a pipeline

### Model

Gradient Boosting Regressor (GBR) used in both pipelines:
- **Scikit-learn**: `GradientBoostingRegressor`
- **PySpark**: `GBTRegressor`

---

## Evaluation Metrics

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

---

## Results

| Metric | Pandas + Sklearn | PySpark MLlib |
|--------|------------------|----------------|
| RMSE   | 4998.02          | 4123.92        |
| MAE    | 2947.88          | 2393.94        |
| R²     | 0.9515           | 0.9672         |

PySpark outperformed the traditional method across all metrics, showcasing efficient handling of categorical features and pipeline scalability.

---

## Key Takeaways

- PySpark’s Pipeline API simplifies preprocessing and modeling at scale
- PySpark provides better performance even on medium-sized datasets
- Data handling (encoding, feature vectorization, etc.) significantly impacts final model performance