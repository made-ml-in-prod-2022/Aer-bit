import os
import pandas as pd
import click
import pickle
import json

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@click.command("validate")
@click.option("--input-dir")
@click.option("--model-dir")
def validate(input_dir: str, model_dir: str) -> None:
    data = pd.read_csv(os.path.join(input_dir, "data_test.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target_test.csv"))

    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(data)
    metrics = {'Accuracy': accuracy_score(target, predictions),
               'F1': f1_score(target, predictions),
               'ROC_AUC': roc_auc_score(target, predictions)}

    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    validate()
