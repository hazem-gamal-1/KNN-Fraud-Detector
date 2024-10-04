
# Credit Card Fraud Detector

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Requirements](#requirements)

## Project Overview
The Credit Card Fraud Detector is a machine learning project that aims to identify fraudulent transactions using the K-Nearest Neighbors (KNN) algorithm. The model leverages various data preprocessing techniques and sampling methods to enhance prediction accuracy. This project involves exploratory data analysis, model training, evaluation, and result logging.

## Features
- Data preparation and preprocessing using Min-Max scaling or Standard scaling.
- Sampling techniques: Random sampling and K-means clustering for handling imbalanced datasets.
- Evaluation metrics: F1 score, precision, and recall for model performance assessment.
- Configurable hyperparameters for model training via command-line arguments.

## Exploratory Data Analysis (EDA) 📊
The EDA notebook (Credit Card Fraud Detection EDA.ipynb) provides a comprehensive analysis of the dataset. Key components include:

- **Data Visualization:** Insights into the distribution of features, correlation matrices.
- **Descriptive Statistics:** Summary statistics that highlight key characteristics of the data.
- **Class Distribution:** Analyzes the proportion of legitimate vs. fraudulent transactions to assess class imbalance.

This analysis helps in understanding the dataset better and making informed decisions during the modeling process.

## Project Structure
```
📂 best_model.json                  # JSON file containing the configuration of the best model
📓 Credit Card Fraud Detection EDA.ipynb   # Jupyter notebook for exploratory data analysis
🔧 credit_fraud_utils_.py          # Utility functions for data preparation, model training, and evaluation
🚀 main.py                          # Main script for executing the KNN model
📊 test.csv                         # Test dataset
📈 train.csv                        # Training dataset
```

## Usage
To run the model, use the command line and specify your desired configurations as arguments. The available arguments include:

- `--dataset`: Path to the training dataset (default: `train.csv`)
- `--preprocessing`: Preprocessing method (0: No preprocessing, 1: MinMaxScaler, 2: StandardScaler, default: 2)
- `--sampling`: Sampling method (1: Random sampling, 2: K-means centers, default: 1)
- `--n_neighbors`: Number of neighbors for KNN (default: 21)

### Example Command
```bash
python main.py --dataset train.csv --preprocessing 1 --sampling 2 --n_neighbors 15
```

## Results
After execution, the configuration settings and evaluation metrics (F1 score, precision, recall, inference time) will be saved in the `config_and_results.json` file.

the **`best_model.json`** file contain the configuration of the best-performing model based on the evaluation metrics.

