import logging 
from typing import Tuple
from typing_extensions import Annotated
import mlflow

import pandas as pd
import numpy as np
from zenml import step 
from src.evaluation import MSE, RMSE, R2_score
from sklearn.base import RegressorMixin

from zenml.client import Client



@step(experiment_tracker="mlflow_tracker_customer")
def evaluate_model(model: RegressorMixin, 
                   X_test: pd.DataFrame,
                   y_test: pd.Series
) -> Tuple[
    Annotated[float, "r2_value"],
    Annotated[float, "rmse_value"]
]:
    """
    Evaluates the model on test set for certain metrics
    """
    try:
        preds = model.predict(X_test)
        mse_value = MSE().calculate_scores(y_test, preds)
        mlflow.log_metric("mse", mse_value)
        r2_value = R2_score().calculate_scores(y_test, preds)
        mlflow.log_metric("r2_score", r2_value)
        rmse_value = RMSE().calculate_scores(y_test, preds)
        mlflow.log_metric("rmse", rmse_value)
        logging.info("Evaluation metrics calculated successfully")
        return r2_value, rmse_value
    
    except Exception as e:
        logging.error(f"Error while evaluating the model: {e}")
        raise e