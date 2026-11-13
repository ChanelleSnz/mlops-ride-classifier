"""Print the version number of the model that has the champion alias (empty if none)."""

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from ride_classifier.train import CHAMPION, MODEL_NAME

try:
    print(MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION).version)
except MlflowException:
    print("")
