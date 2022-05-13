ML project
==============================

Installation:

~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~
-------------------------------

Generate train and test corpus from raw data:
~~~
python ml_project/split.py -cf configs/config.yaml
~~~

Run training pipeline:
~~~
python ml_project/train.py -cf configs/config.yaml
~~~

Generate predictions:
~~~
python ml_project/predict.py -cf configs/config.yaml
~~~
-------------------------------
Tests:
~~~
py.test -v tests/tests.py
~~~

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── configs            <- The top-level configs for ml specialists experimenting with their models. 
    ├── data
    │   ├── heart_cleveland_upload.csv.dvc       <- The original, immutable data dump.
    │   ├── train.csv.dvc                        <- Train corpus (obtained using split.py).
    │   ├── test.csv.dvc                         <- Validation corpus (obtained using split.py).
    │   └── predictions.csv.dvc                  <- Generated predictions.
    │
    ├── ml_project
    │   ├── predict.py                           <- Script for generatining predictions.
    │   ├── split.py                             <- Script to generate train/test corpus from raw data.
    │   ├── train.py                             <- Script to train model.
    │   └── utils.py                             <- Module with helper functions.
    │
    ├── models             <- Trained models.
    │
    ├── notebooks          <- Jupyter notebooks with EDA.
    │
    ├── tests
    │   ├── test_dta                             <- Folder with test output files.
    │   ├── conftest.py                          <- Pytest configs.
    │   ├── tests.py                             <- Module with tests for train and prediction pipelines.
    |
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment.
    │                 
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
=======
