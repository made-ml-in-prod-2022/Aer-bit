import os
import pytest


@pytest.fixture()
def config_path():
    return 'configs/config.yaml'


@pytest.fixture()
def input_data_path():
    return 'data/heart_cleveland_upload.csv'


@pytest.fixture()
def train_data_path():
    return 'data/train.csv'


@pytest.fixture()
def test_data_path():
    return 'data/test.csv'


@pytest.fixture()
def target_col():
    return ['condition']


@pytest.fixture()
def feature_cols():
    return ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']


@pytest.fixture()
def generated_train_data_path():
    return 'tests/test_data/train.csv'


@pytest.fixture()
def generated_test_data_path():
    return 'tests/test_data/test.csv'


@pytest.fixture()
def test_model_path():
    return 'tests/test_data/model.pkl'


@pytest.fixture()
def test_predictions_path():
    return 'tests/test_data/predictions.csv'