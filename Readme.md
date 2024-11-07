## KNN Fraud Detector

### Table of Contents
- [Project Overview](#project-overview)
- [Key Insights](#key-insights) 
- [Sampling Techniques](#sampling-techniques)
- [Modeling](#modeling)
- [Results](#results)

---

### Project Overview
The **KNN Fraud Detector** is a machine learning project designed to detect fraudulent credit card transactions using the K-Nearest Neighbors (KNN) algorithm. Given the highly imbalanced nature of fraud detection datasets, the project implements specific techniques for sampling and preprocessing to improve model performance. The goal is to build an effective classifier that can predict fraudulent transactions based on historical transaction data. The project steps include data preparation, feature scaling, model training, and performance evaluation.
---


## Key Insights 

These insights are derived from the Exploratory Data Analysis (EDA) conducted on the dataset:

1. **Class Imbalance**: The dataset reveals a significant imbalance between fraudulent and non-fraudulent transactions, necessitating techniques like oversampling or undersampling to improve model performance.

2. **Feature Distributions**: Several numerical features, particularly `Amount`, exhibit skewness and outliers, which could impact predictive modeling.

3. **Correlations**: The heatmap indicates that certain features have strong correlations with the target variable, highlighting potential predictors of fraud that should be prioritized in model training.

4. **Fraud Patterns**: Visualizations suggest identifiable patterns in fraudulent transactions, such as specific transaction amounts, which can inform targeted fraud detection strategies.


---

### Sampling Techniques
To handle the class imbalance between fraudulent and non-fraudulent transactions, the following sampling techniques were applied:

1. **Random Sampling:** A simple method that randomly selects negative (non-fraud) instances.
2. **K-means Clustering:** This technique clusters negative instances and uses the cluster centers to balance the dataset.

For this project, **K-means centers** were used to represent the majority class effectively.

---

### Modeling
The KNN model was trained with **71 neighbors**, a value chosen based on experimentation with the dataset. This model classifies fraudulent transactions by finding the nearest neighbors in the feature space and labeling the transaction accordingly.

---

### Results

| Metric        | Value          |
|---------------|----------------|
| **F1 Score**  | 0.773          |
| **AUC-PR**    | 0.600          |
| **Neighbors** | 71             |
| **Sampling**  | K-means centers|
| **Preprocessing** | MinMaxScaler|
