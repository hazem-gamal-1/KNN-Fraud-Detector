import time
import json
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score,precision_score,recall_score,average_precision_score
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
def prepare_and_load_data(path):
    df=pd.read_csv(path)
    df.drop(columns='time', inplace=True)
    df.columns = map(str.lower, df.columns)
    df.rename(columns={'class': 'label'}, inplace=True)
    df['amount'] = np.log1p(df.amount)  # others looks scaled
    X = df[df.columns[:-1].tolist()]
    y = df[df.columns[-1]]
    return X, y


def sample_data(X_train, Y_train, sampling_method):
    X_pos = X_train[Y_train == 1]
    Y_pos = Y_train[Y_train == 1]
    X_neg = X_train[Y_train == 0]
    Y_neg = Y_train[Y_train == 0]

    if sampling_method == 1:
        # Randomly sample negative instances
        X_neg_indices = np.random.choice(len(X_neg),int(len(Y_pos*1.5)), replace=False)
        X_neg = X_neg.iloc[X_neg_indices]
        Y_neg = Y_neg.iloc[X_neg_indices]
    else:
        # Use K-means to get centers of negative instances
        kmeans = KMeans(n_clusters=int(len(Y_pos)*1.5), random_state=42)
        kmeans.fit(X_neg)
        X_neg = kmeans.cluster_centers_
        Y_neg = np.zeros(len(X_neg))  # Label negative samples as 0

    # Combine positive and sampled negative data
    X = np.vstack([X_neg, X_pos])
    Y = np.concatenate([Y_neg, Y_pos])
    return X, Y


def Get_preprocessor(option):
    if option==1:
        return MinMaxScaler()
    elif option==2:
        return  StandardScaler()

    return  None



def train_model(X, Y, n_neighbors):
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X, Y)
    return knn_model



def evaluate_model(model, X_test, Y_test,type):
    start_time = time.time()
    y_pred = model.predict(X_test)
    f1 = f1_score(Y_test, y_pred)
    auc_pr=average_precision_score(Y_test,y_pred)
    precision = precision_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred)
    inference_time = time.time() - start_time

    print()
    print(f"{type}_metrics")
    print(f'KNN F1 Score: {f1 * 100:.2f}% | '
          f'Precision: {precision * 100:.2f}% | '
          f'Recall: {recall * 100:.2f}%')
    print(f"Auc_PR: {auc_pr}")
    print(f'Inference Time: {inference_time:.2f} seconds')

    return {
        "f1_score": f1,
        "Auc_PR":auc_pr
    }





def save_config_and_results(args,results_testing, path):
    config = {
        "preprocessing": "MinMaxScaler" if args.preprocessing == 1 else "StandardScaler" if args.preprocessing == 2 else "No preprocessing",
        "sampling": "Random sampling" if args.sampling == 1 else "K-means centers",
        "n_neighbors": args.n_neighbors
    }

    data = {
        "config": config,
        "results_testing":results_testing
    }

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)