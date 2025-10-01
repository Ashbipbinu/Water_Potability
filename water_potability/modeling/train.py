import os
import pandas  as pd
import pickle

from sklearn.ensemble import RandomForestClassifier

def load_data(file_path):
    return pd.read_csv(file_path)

def splitting_data_to_XY(data: pd.DataFrame):
    X = data.drop(columns=['Potability'], axis=1)
    y = data['Potability']

    return (X, y)

def model_training(model, X_data, y_data):
    model.fit(X_data, y_data)

    return model

def saving_model(model):

    directory = os.path.join(os.getcwd(), 'models')
    os.makedirs(directory, exist_ok=True)

    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)


def main():

    print("Loading the data")
    file_path = os.path.join(os.getcwd(), 'data', 'processed', 'train_processed.csv')
    train_data = load_data(file_path)

    print("Splitting the data into features and target")
    X_train, y_train = splitting_data_to_XY(train_data)

    rf = RandomForestClassifier(n_estimators=100, max_depth=3)

    print("Model training started")   
    model = model_training(rf,X_train, y_train)   

    print("Saving the model") 
    saving_model(model)


if __name__ == "__main__":
    main()

    print("Finished training")