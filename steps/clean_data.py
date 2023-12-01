import logging 
import pandas as pd
from zenml import step 
from typing import Tuple
from typing_extensions import Annotated

from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy


@step 
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """
    pre-processing the dataset and dividing into train and test sets

    Args:
        df: raw data
    Returns:
        X_train: Training features
        X_test: Test features
        y_train: Training labels 
        y_test: Test labels
    """
    try:
        # pre processing of the dataset
        process_strategy = DataPreProcessStrategy()
        df_cleaning = DataCleaning(df, process_strategy)
        processed_data = df_cleaning.handle_data() 

        # splitting of the dataset
        divide_strategy = DataDivideStrategy()
        df_splitting = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = df_splitting.handle_data()
        logging.info("Data Cleaning Completed")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error(f"Error in data cleaning step of the pipeline: {e}")
        raise e




