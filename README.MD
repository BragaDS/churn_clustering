# Churn Prediction Project

## Overview

This project focuses on predicting customer churn using machine learning techniques. The dataset includes features related to customer behavior, and the goal is to build a model that can effectively identify customers likely to churn.

## Project Structure

The project is organized into the following key phases:

- **Data Exploration and Analysis:** Conducting an initial exploration and analysis of the dataset to gain insights into its characteristics.

- **Data Pre-processing:** Addressing missing values, outliers, and applying necessary transformations to prepare the data for modeling.

- **Cluster Analysis:** Utilizing unsupervised learning techniques, such as KMeans clustering, to identify distinct customer groups.

- **Modeling:** Developing, training, and evaluating machine learning models for churn prediction.

- **Evaluation:** Comparing the performance of different models and identifying the best-performing one.

- **Conclusion:** Summarizing key findings, observations, and recommendations based on the project analysis.

## Technologies Used

The project utilizes the following technologies:

- Python
- Jupyter Notebooks
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Imbalanced-learn, etc.

## Data Analysis and Cluster Analysis

The data analysis phase involves examining the relationships between various features and churn rates. Additionally, cluster analysis was performed to identify distinct customer groups based on their behavior and characteristics.

### Insights from Cluster Analysis:

- **Products Number:** Customers with a higher number of products exhibit a lower likelihood of cancellation.

- **Credit Card:** The presence of a credit card may have a slightly positive influence on customer retention.

- **Active Members:** Being an active member is correlated with a lower probability of cancellation.

- **Country:** Customers from different countries exhibit varying churn rates, with Country 1 having the highest churn rate.

- **Tenure:** No clear trend in cancellation rate based on tenure.

These insights provide an initial understanding of the relationships between variables and churn rates, guiding further analysis.

## Results

The final selected model is XGBoost, achieving the following metrics:

- **Accuracy:** 87%
- **Precision (Class 0):** 88%
- **Recall (Class 0):** 97%
- **F1-score (Class 0):** 92%
- **Precision (Class 1):** 80%
- **Recall (Class 1):** 46%
- **F1-score (Class 1):** 58%

## Conclusion and Recommendations

- **Model Performance:** The XGBoost model outperformed the Random Forest model, particularly in terms of recall for Class 1.

- **Data Processing Surprises:** Surprisingly, the models performed better without outlier removal and pre-processing, suggesting that the original data distribution contains valuable information.

- **Cluster Analysis:** The cluster analysis revealed insights into different customer groups, providing a richer understanding of factors influencing churn.

- **Future Considerations:** Further fine-tuning of hyperparameters and exploration of feature engineering techniques could be considered for potential performance improvement.
