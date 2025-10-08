import mlflow
import pickle
import yaml
import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV




#dagshub.init(repo_owner='Ashbipbinu', repo_name='Water_Potability', mlflow=True)
new_Experiment = "Water_Potability_Classification-Auto"

if mlflow.get_experiment_by_name(new_Experiment) == None:
    experiment_id = mlflow.create_experiment(name = new_Experiment)

mlflow.set_experiment(new_Experiment)

#mlflow.set_tracking_uri("https://dagshub.com/Ashbipbinu/Water_Potability.mlflow") 
mlflow.set_tracking_uri("http://127.0.0.1:5000") 

params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'params.yaml')
with open(params_path, 'r') as file:
        params = yaml.safe_load(file)

n_estimators = params['model_training']['n_estimators']
max_depth = params['model_training']['max_depth']
clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

param_dist = {
        "n_estimators": [100, 200, 1000],
        "max_depth" : [10, 20, 30, 40, 50, None],
        "max_features" : ['sqrt', 'log2', None],
        "min_samples_split" : [2, 5, 10]
    }
random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_dist, n_iter=100, cv=2, verbose=2, random_state=42, n_jobs=-1)

with mlflow.start_run(run_name="Random Forest Tuning") as parent: # Need to log each parameter in each cv, so parent run is initiated and inside which child run will be done


# Importing train data its processing

    params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'params.yaml')
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)

    def load_data(file_path):
        return pd.read_csv(file_path)

    def splitting_data_to_XY(data: pd.DataFrame):
        X = data.drop(columns=['Potability'], axis=1)
        y = data['Potability']

        return (X, y)

    def model_training(model, X_data, y_data):
        model.fit(X_data, y_data)
        return model

    file_path = os.path.join(os.getcwd(), "data", "processed", "train_processed_mean.csv")
    load_train_data = load_data(file_path)

    split_data = splitting_data_to_XY(load_train_data)
    X_train, y_train = split_data
   

    model = model_training(clf, X_train, y_train)
    

    def load_data(file_path):
        return pd.read_csv(file_path)


    def make_prdiction(model, X_data):
        y_pred = model.predict(X_data)
        return y_pred

    def evaluation(y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision_score": precision,
            "recall_score": recall
        }
    

    
        

    random_search.fit(X_train, y_train)

    print(random_search.best_params_)

    params = random_search.best_params_

    best_estimator = random_search.best_estimator_
    with open("best_estimator.pkl", 'wb') as file:
        pickle.dump(best_estimator, file)

    for i in range(len(random_search.cv_results_['params'])):
        with mlflow.start_run(run_name=f"Combination {i+1}:", nested=True) as child:
            mlflow.log_params(random_search.cv_results_['params'][i])
            mlflow.log_metric("mean_test_score",random_search.cv_results_['mean_test_score'][i])

    # Importing test data its processing

    file_path = os.path.join(os.getcwd(), "data", "processed", "test_processed_mean.csv")
    load_test_data = load_data(file_path)
    X_test, y_test = splitting_data_to_XY(load_test_data)
    y_pred = make_prdiction(best_estimator, X_test)

    metrics = evaluation(y_test, y_pred)

    train_df = mlflow.data.from_pandas(load_train_data)
    test_df = mlflow.data.from_pandas(load_test_data)

    accuracy = metrics['accuracy']
    f1_score = metrics['f1_score']
    precision_score = metrics['precision_score']
    recall_score = metrics['recall_score']

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    with open("best_estimator.pkl", 'rb') as file:
        model = pickle.load(file)
        mlflow.sklearn.log_model(sk_model=model, 
        artifact_path="best_model")

    mlflow.log_artifact(__file__)
