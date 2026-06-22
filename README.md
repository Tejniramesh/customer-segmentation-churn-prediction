# Customer Segmentation and Churn Prediction

## Project Overview

This project focuses on analyzing customer behavior, segmenting customers into meaningful groups, and predicting customer churn using Machine Learning techniques.

## Dataset

**Dataset:** Telco Customer Churn Dataset

* Total Records: 7043
* Total Features: 21
* Source: Kaggle

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* Git
* GitHub
* VS Code

## Week 1 Tasks Completed

### 1. Exploratory Data Analysis (EDA)

* Loaded and explored the dataset
* Checked data types and dataset structure
* Analyzed dataset dimensions
* Examined churn distribution

### 2. Data Cleaning

* Converted TotalCharges to numeric format
* Removed missing values
* Prepared data for Machine Learning

### 3. Customer Segmentation using K-Means

* Selected key features:

  * Tenure
  * MonthlyCharges
  * TotalCharges
* Scaled features using StandardScaler
* Applied K-Means Clustering (3 segments)
* Segmented customers into different groups

### 4. Basic Classification using Logistic Regression

* Converted Churn into binary values
* Split dataset into training and testing sets
* Trained Logistic Regression model
* Evaluated model performance

## Results

### Customer Segmentation

* Segment 0: 2215 customers
* Segment 1: 3346 customers
* Segment 2: 1471 customers

### Classification Performance

* Logistic Regression Accuracy: 77.97%

## Project Structure

customer-segmentation-churn-prediction/

├── data/

│ └── WA_Fn-UseC_-Telco-Customer-Churn.csv

├── notebooks/

│ ├── eda.py

│ ├── week1_kmeans.py

│ └── basic_classification.py

└── README.md

## Future Enhancements

* Decision Tree Classification
* Random Forest Classification
* Model Comparison
* Advanced Feature Engineering
* Interactive Dashboard
* Model Deployment
## Week 2 Progress

- Performed Customer Segmentation using K-Means Clustering
- Created 3 customer segments:
  - New Customers
  - Premium Customers
  - Regular Customers
- Analyzed customer behavior based on tenure, monthly charges and total charges
- Generated customer segment distribution graph
- Performed churn analysis for each customer segment
- Generated churn rate graph by segment
- Exported segmented customer dataset (customer_segments.csv)