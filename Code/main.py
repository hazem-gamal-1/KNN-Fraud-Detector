import argparse
from credit_fraud_utils_ import *
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fraud Detector using KNN')
    parser.add_argument('--dataset', type=str,
                        default=r"E:\ML homework\Projects\project 3 Fraud Detection with KNN\KNN-Fraud-Detector\Code\Data\train.csv",
                        help='Path to the training dataset')

    parser.add_argument('--preprocessing', type=int, default=1, choices=[1, 2],
                        help='''Preprocessing method:
                                    0: No preprocessing
                                    1: MinMaxScaler 
                                    2: StandardScaler''')
    parser.add_argument('--sampling', type=int, default=2, choices=[1, 2],
                        help='''Sampling method:
                                    1: Random sampling
                                    2: K-means centers''')
    parser.add_argument('--n_neighbors', type=int, default=71,
                        help='Number of neighbors for KNN')

    args = parser.parse_args()

    # Step 1: Load and prepare the data
    X_train, Y_train = prepare_and_load_data(args.dataset)
    X_test, Y_test = prepare_and_load_data(r"E:\ML homework\Projects\project 3 Fraud Detection with KNN\KNN-Fraud-Detector\Code\Data\test.csv")

    # Step 2: Sampling
    X, Y = sample_data(X_train, Y_train,args.sampling)

    # Step 3: Preprocessing
    preprocessor = Get_preprocessor(args.preprocessing)
    if preprocessor is not None:
        X = preprocessor.fit_transform(X)
        X_test = preprocessor.transform(X_test)

    # Step 4: Modeling
    model = train_model(X, Y, args.n_neighbors)

    # Step 5: Evaluation
    testing_metrics=evaluate_model(model,X_test,Y_test,"Testing")

    # step 6: Saving config andresults
    path=r"E:\ML homework\Projects\project 3 Fraud Detection with KNN\KNN-Fraud-Detector\config_and_results.json"
    save_config_and_results(args,testing_metrics,path)