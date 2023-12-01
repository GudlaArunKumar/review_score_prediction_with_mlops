import logging 
import pandas as pd
from zenml import step

class IngestData:
    """
    Ingesting data from data path
    """
    def __init__(self):
        """
        Instaniting the class 
        """
        self.data_path = "/mnt/e/Machine_Learning_Projects/ecommerce_review_score_prediction/data/olist_customers_dataset.csv"

    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path) 

@step
def ingest_df() -> pd.DataFrame:
    """
    Ingesting the data from the data path 

    Args:
        data_path: path to the data
    Returns:
        pd.DataFrame: The ingested data
    """
    try:
        ing_data = IngestData()
        df = ing_data.get_data()
        return df 
    except Exception as e:
        logging.error(f"Error while Ingesting data: {e}")
        raise e