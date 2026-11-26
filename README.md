# RainfallGuard Kerala

RainfallGuard Kerala is a data-driven rainfall analysis and prediction project built using PySpark and Machine Learning. The project analyses historical rainfall records across stations in Kerala to identify rainfall patterns, detect statistical anomalies, and predict next-day rainfall amounts and heavy rainfall events.

The system combines distributed data processing, exploratory analysis, statistical anomaly detection, feature engineering, and machine learning to explore rainfall behaviour and predictive modelling using historical station-level data.

## Key Features

- **Data Processing & Cleaning:** Validates, cleans, and processes rainfall observations using PySpark.
- **Rainfall Analysis:** Analyses rainfall trends across stations, districts, months, and seasons using Spark SQL.
- **Anomaly Detection:** Identifies unusual rainfall observations using statistical techniques, including percentile-based thresholds and Z-scores.
- **Feature Engineering:** Generates lag features, rolling rainfall statistics, temporal features, and station-related information for prediction.
- **Rainfall Regression:** Predicts next-day rainfall amounts using Random Forest and Gradient Boosted Tree regressors.
- **Heavy Rainfall Classification:** Classifies potential heavy rainfall events using Random Forest, Gradient Boosted Trees, and Logistic Regression.
- **Model Evaluation:** Evaluates models using regression and classification metrics, validation-based model selection, and baseline comparisons.
- **Visualisation & Reporting:** Generates rainfall analysis plots, model evaluation charts, anomaly reports, and prediction outputs.

## Technologies Used

- Python
- Apache Spark / PySpark
- Spark SQL
- Spark MLlib
- Pandas
- Matplotlib
- Seaborn

## Project Objective

The objective of RainfallGuard Kerala is to explore how large-scale data processing, statistical analysis, and machine learning can be combined to analyse historical rainfall data and develop predictive models for rainfall estimation and heavy rainfall identification.

This project is an experimental historical-data-based prediction system and is not intended to serve as a real-time meteorological forecasting or flood-warning system.
