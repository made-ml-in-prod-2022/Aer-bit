import os
import pandas as pd
import click
import pickle
import joblib


@click.command("predict")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--model-dir")
def predict(input_dir: str, output_dir: str, model_dir: str) -> None:
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))

    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    pipe = joblib.load(os.path.join(model_dir, "preprocessing_pipeline.gz"))
    data_transformed = pipe.transform(data)

    os.makedirs(output_dir, exist_ok=True)
    predictions = pd.Series(model.predict(data_transformed))
    predictions.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)


if __name__ == '__main__':
    predict()
