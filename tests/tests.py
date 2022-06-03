import os
import pickle
import numpy as np
import pandas as pd
from hydra import initialize, compose
from sklearn.utils.validation import check_is_fitted

from ml_project.utils import generate_dataset
from ml_project.train import train_pipeline
from ml_project.predict import prediction_pipeline

    

def test_dataset(input_data_path: str,
                 target_col: list,
                 feature_cols: list) -> None:
    
    data = pd.read_csv(input_data_path)
    
    assert len(data) > 10
    assert target_col[0] in data.columns
    assert np.all(x.isdigit() for x in data[target_col])
    
    for col in feature_cols:
        assert col in data.columns
        assert np.all(x.isnumeric() for x in data[col])


def test_split_data(train_data_path: str,
                    test_data_path: str) -> None:
    
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    
    assert len(train_data) > 10
    assert len(test_data) > 10
    assert len(train_data) + len(test_data) > len(train_data)
    assert len(train_data) + len(test_data) > len(test_data)
    

def test_generate_dataset(generated_train_data_path: str,
                          generated_test_data_path: str,
                          feature_cols: list,
                          target_col: list) -> None:

    generate_dataset(generated_train_data_path, 
                     generated_test_data_path, 
                     feature_cols,
                     target_col)
    
    assert os.path.exists(generated_train_data_path)
    assert os.path.exists(generated_test_data_path)
    
    
def test_train_pipeline(generated_train_data_path: str,
                        test_model_path: str) -> None:
    
    with initialize(config_path='../ml_project/configs'):
        configs = compose(config_name='config',  overrides=['hydra.run.dir=../ml_project/tests/'])
        configs.train_data_path = generated_train_data_path
        configs.output_model_path = test_model_path
        train_pipeline(configs)
        
    assert os.path.exists(test_model_path)
    
    model = pickle.load(open(test_model_path, "rb"))
    check_is_fitted(model)

    
def test_prediction_pipeline(generated_test_data_path: str,
                             test_model_path: str,
                             test_predictions_path: str) -> None:
    
    with initialize(config_path='../ml_project/configs'):
        configs = compose(config_name='config',  overrides=['hydra.run.dir=../ml_project/tests/'])
        configs.test_data_path = generated_test_data_path
        configs.output_model_path = test_model_path
        configs.output_data_path = test_predictions_path
        prediction_pipeline(configs)
    
    
    assert os.path.exists(test_predictions_path)
    
    