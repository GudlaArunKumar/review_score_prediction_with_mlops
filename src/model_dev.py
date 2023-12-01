import logging
from abc import ABC, abstractmethod  

from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class for all models
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Training the model 
        """
        pass 


class LinearRegressionModel(Model):
    """
    Training the data with Linear regression model
      with respective hyper parameters
    """

    def train(self, X_train, y_train, **kwargs) -> LinearRegression:
        
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Linear Regression model trained suuccefully")
            return reg
    
        except Exception as e:
            logging.error(f"Error while training linear regression model: {e}")
            raise e 




