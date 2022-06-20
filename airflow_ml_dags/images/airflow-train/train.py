import os
import pandas as pd
import click
import pickle

from sklearn.linear_model import LogisticRegression


@click.command("train")
@click.option("--input-dir")
@click.option("--model-dir")
def train(input_dir: str, model_dir: str) -> None:
    data = pd.read_csv(os.path.join(input_dir, "data_train.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target_train.csv"))

    model = LogisticRegression().fit(data, target)
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    train()
