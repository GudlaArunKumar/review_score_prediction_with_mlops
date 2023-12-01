from pipelines.training_pipeline import train_pipeline 


if __name__ == "__main__":
    my_pipeline = train_pipeline.with_options(enable_cache=False)
    my_pipeline()