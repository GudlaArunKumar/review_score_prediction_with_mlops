import logging 
import pandas as pd
from zenml import step 
import mlflow

from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

from zenml.client import Client

#experiment_tracker = Client().activate_stack.experiment_tracker

@step(experiment_tracker="mlflow_tracker_customer")
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: ModelNameConfig
) -> RegressorMixin:
    
    """
    Trains the model on ingested data 
    """
    try:
        model = None 
        if config.model_name == "LinearRegression":
            # logging models and metrics
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model Name not suppported {config.model_name}")
    except Exception as e:
        logging.error("Error while training the model: {}".format(e))
        raise e

    
