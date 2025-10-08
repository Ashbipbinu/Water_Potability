import mlflow

from mlflow.tracking import MlflowClient

client = MlflowClient()

model_name = "Best Model"
new_stage = 'Staging'
model_version = 2


client.transition_model_version_stage(
    archive_existing_versions=False,
    name=model_name,
    stage=new_stage,
    version=model_version
)