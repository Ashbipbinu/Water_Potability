import mlflow

from mlflow.tracking import MlflowClient

client = MlflowClient()


run_id = '7930fc7ca8b24298bb9b28732e72524a'
model_path = "file:///F:/Water_Potability/mlruns/179076651016105839/models/m-00c4def450c143eaa91383cc9ac299d5/artifacts/MLmodel"
model_name =  "Best Model"


model_uri = f'run://{run_id}/{model_path}'

reg = mlflow.register_model(model_uri, model_name)