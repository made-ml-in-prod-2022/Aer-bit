import os
import pandas as pd
import click
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--model-dir")
def preprocess(input_dir: str, output_dir: str, model_dir: str) -> None:
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))

    features = data.columns
    pipe = Pipeline([('imputer', SimpleImputer()), ('scaler', StandardScaler())])
    data_transformed = pipe.fit_transform(data)
    data_transformed = pd.DataFrame(data_transformed, columns=features)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(pipe, os.path.join(model_dir, "preprocessing_pipeline.gz"))

    os.makedirs(output_dir, exist_ok=True)
    data_transformed.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    target.to_csv(os.path.join(output_dir, "target.csv"), index=False)


if __name__ == '__main__':
    preprocess()
