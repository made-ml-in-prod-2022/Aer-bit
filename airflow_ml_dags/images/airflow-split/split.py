import os
import pandas as pd
import click

from sklearn.model_selection import train_test_split


@click.command("split")
@click.option("--input-dir")
@click.option("--output-dir")
def split(input_dir: str, output_dir: str) -> None:
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

    for name, df in {'data_train': X_train, 'data_test': X_test, 'target_train': y_train, 'target_test': y_test}.items():
        df.to_csv(os.path.join(output_dir, "{}.csv".format(name)), index=False)


if __name__ == '__main__':
    split()
