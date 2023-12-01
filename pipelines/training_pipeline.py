from zenml import pipeline 
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.model_evaluation import evaluate_model

@pipeline
def train_pipeline(): #ingest_df, clean_df, train_model, evaluate_model
    df = ingest_df()
    X_train, X_test, y_train, y_test  = clean_df(df)
    model = train_model(X_train, y_train, X_test, y_test)
    r2_score, rmse_score = evaluate_model(model, X_test, y_test)

    print(f"R2_score of the model: {r2_score}")
    print(f"RMSE score of the model: {rmse_score}")

