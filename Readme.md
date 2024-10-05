# KNN Fraud Detector

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)

## Project Overview
The **KNN Fraud Detector** project aims to identify fraudulent transactions using the K-Nearest Neighbors (KNN) algorithm. The project utilizes various data preprocessing and sampling techniques to handle class imbalance and improve model performance.

## Features
- **Data Preparation and Preprocessing:** Utilizes Min-Max scaling or Standard scaling for feature normalization.
- **Sampling Techniques:** Implements random sampling and K-means clustering to manage imbalanced datasets.
- **Evaluation Metrics:** Assesses model performance using F1 score, precision, and recall.
- **Configurable Hyperparameters:** Allows for command-line arguments to set hyperparameters during model training.

## Exploratory Data Analysis (EDA) 📊
The EDA notebook (`Credit Card Fraud Detection EDA.ipynb`) provides a thorough analysis of the dataset. Key components include:

- **Data Visualization:** Insights into the distribution of features and correlation matrices.
- **Descriptive Statistics:** Summary statistics highlighting key characteristics of the data.
- **Class Distribution:** Analysis of the proportion of legitimate vs. fraudulent transactions to assess class imbalance.

This analysis helps in understanding the dataset better and making informed decisions during the modeling process.

## Project Structure
```
📂 best_model.json                             # JSON file containing the configuration of the best model
📓 Credit Card Fraud Detection EDA.ipynb        # Jupyter notebook for exploratory data analysis
🔧 credit_fraud_utils_.py                       # Utility functions for data preparation, model training, and evaluation
🚀 main.py                                     # Main script for executing the KNN model
📊 test.csv                                    # Test dataset
📈 train.csv                                   # Training dataset
📁 val.csv                                     # Validation dataset for model evaluation
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
After execution, the configuration settings and evaluation metrics (F1 score, precision, recall, inference time) will be saved in the `config_and_results.json` file. The **`best_model.json`** file contains the configuration of the best-performing model based on the evaluation metrics.
