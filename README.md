<img src ="Images/banner.gif"/>

# Sales Prediction Model

## Table of Contents
1. [Overview](#overview)
2. [Models Used](#models-used)
3. [Data Preprocessing](#data-preprocessing)
4. [Data Loading](#data-loading)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [Models Training and Evaluation](#models-training-and-evaluation)
   - 6.1 [Model Training](#model-training)
   - 6.2 [Model Evaluation](#model-evaluation)
7. [Interpretation of Report](#interpretation-of-report)
8. [Data Visualization](#data-visualization)
9. [Findings](#findings)
   - 9.1 [Data Exploration](#data-exploration)
   - 9.2 [Model Performance](#model-performance)
10. [Output](#output)
11. [Conclusion](#conclusion)

## 1. Overview
This project aims to predict product sales based on advertising expenditures, focusing on 'TV advertising'. Machine learning techniques are employed to analyze and interpret data, enabling businesses to optimize advertising strategies and maximize sales potential.

## 2. Models Used
- **Random Forest Regression**: Utilized for its ability to handle complex relationships and provide robust predictions.
- **Linear Regression**: Employed as a baseline for comparison due to its simplicity and interpretability.

## 3. Data Preprocessing
Data preprocessing involves:
- Handling missing values (if any).
- Encoding categorical variables (if applicable).
- Scaling numerical features.

## 4. Data Loading
The advertising dataset is loaded from a CSV file containing columns for 'TV', 'Radio', 'Newspaper' advertising expenditures, and 'Sales'.

<img src="https://images.squarespace-cdn.com/content/v1/61f30f2d94a2222a9c7f5389/8c1fe081-054b-4d2e-bae4-bbcbaf320bf2/client-layered.gif" width="400"/>

## 5. Exploratory Data Analysis (EDA)
EDA is performed to:
- Visualize relationships between features and target variable ('Sales').
- Identify correlations and distributions of features.
- Detect outliers or anomalies in the data.

## 6. Models Training and Evaluation
### 6.1. Model Training
1. **Random Forest Regression**:
   - GridSearchCV used to optimize hyperparameters.
   - Best model selected based on cross-validated negative MSE score.

2. **Linear Regression**:
   - Simple model trained as a baseline for comparison.

### 6.2. Model Evaluation
Evaluation metrics computed include:
- **Mean Squared Error (MSE)**: Measure of prediction accuracy.
- **Mean Absolute Error (MAE)**: Provides absolute measure of average error.
- **R-squared (R2)**: Indicates goodness of fit of the model.

## 7. Interpretation of Report
- Comparison of model performance based on evaluation metrics.
- Analysis of coefficients (for Linear Regression) and feature importances (for Random Forest) to interpret relationships between 'TV advertising' and 'Sales'.

### Output Interpretation and Explanation
- **Random Forest Model:**
  - **Best Parameters:** {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
  - **Best Negative MSE Score:** -1.6148
  - **Evaluation Metrics on Test Set (Random Forest):**
    - **Mean Squared Error (MSE):** 1.4591
    - **Mean Absolute Error (MAE):** 0.9170
    - **R-squared (R2):** 0.9528

- **Linear Regression Model:**
  - **Coefficients:** 0.0555
  - **Intercept:** 7.0071
  - **Evaluation Metrics on Test Set (Linear Regression):**
    - **Mean Squared Error (MSE):** 6.1011
    - **R-squared (R2):** 0.8026

#### Explanation in Non-Technical Terms
- **Model Comparison:** The Random Forest model performs better than the Linear Regression model in predicting sales based on TV advertising. It achieves this by considering complex interactions and nonlinear relationships in the data, leading to more accurate predictions.
  
- **Interpretation:** For businesses, these models provide insights into how changes in advertising spending (specifically on TV) can impact sales. They help optimize advertising budgets by predicting potential sales outcomes with different strategies.

- **Conclusion:** Based on these results, businesses can use the Random Forest model to make more reliable predictions about the effectiveness of their advertising campaigns, thereby maximizing their sales potential.

## 8. Data Visualization
- Visual representations include scatter plots, pair plots, and bar plots to illustrate relationships and distributions.
- Plots of model predictions vs. actual sales to assess performance visually.
  
<img width="1312" alt="Screenshot 2024-06-15 at 01 30 28" src="https://github.com/noturlee/Sales-DataAnalysis-CODSOFT/assets/100778149/53a4a8d2-922d-4494-942b-5839c35d22a4">

## 9. Findings
### 9.1. Data Exploration
- Strong positive correlation observed between 'TV advertising' and 'Sales'.
- 'TV' expenditure shows highest influence on 'Sales' compared to 'Radio' and 'Newspaper'.

### 9.2. Model Performance
- **Random Forest** outperforms Linear Regression in terms of predictive accuracy.
- Lower MSE and higher R-squared indicate Random Forest captures the relationship more effectively.

## 10. Output
- Predicted sales values for new data points using both Random Forest and Linear Regression models.
  
<img width="1277" alt="Screenshot 2024-06-15 at 01 31 07" src="https://github.com/noturlee/Sales-DataAnalysis-CODSOFT/assets/100778149/f8d3bc76-d218-424a-bd21-583c0fb75caf">

## 11. Conclusion
- **Random Forest** is recommended for predicting sales based on 'TV advertising' due to its superior performance.
- Insights gained can guide advertising strategies to optimize spending and maximize sales.
