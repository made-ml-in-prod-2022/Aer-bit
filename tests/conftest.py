import os
import pytest


@pytest.fixture()
def config_path():
    return 'configs/config.yaml'


@pytest.fixture()
def data_path():
    return 'data/'


@pytest.fixture()
def data_filename():
    return 'heart_cleveland_upload.csv'


@pytest.fixture()
def train_filename():
    return 'train.csv'


@pytest.fixture()
def test_filename():
    return 'test.csv'


@pytest.fixture()
def target_col():
    return ['condition']


@pytest.fixture()
def feature_cols():
    return ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']


@pytest.fixture()
def test_data_path():
    return 'tests/test_data/'


@pytest.fixture()
def test_model_path():
    return 'tests/test_data/model.pkl'


@pytest.fixture()
def test_predictions_path():
    return 'tests/test_data/predictions.csv'