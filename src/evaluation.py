import logging 
from abc import ABC, abstractmethod 

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluating the models
    """

    @abstractmethod 
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculated the evaluation metrics of the model
        """
        pass 

class MSE(Evaluation):
    """
    Evaluation strategy for calculating Mean Squared Error
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            mse = mean_squared_error(y_true, y_pred)
            return mse
            logging.info(f"MSE calculated: {mse}")
        except Exception as e:
            logging.error(f"Error while calculating MSE: {e}")
            raise e
        
class R2_score(Evaluation):
    """
    Evaluation strategy for calculating r-squared error
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            r2_value = r2_score(y_true, y_pred)
            logging.info(f"R2 score calculated: {r2_value}")
            return r2_value
        except Exception as e:
            logging.error(f"Error while calculating R2 score: {e}")
            raise e

class RMSE(Evaluation):
    """
    Evaluation strategy for calculating Root Mean Squared Error
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info(f"RMSE Score calculated: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error while calculating RRMSE score: {e}")
            raise e