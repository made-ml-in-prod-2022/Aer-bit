import os
import pytest
import numpy as np
import pandas as pd
import ml_project
from ml_project.utils import get_configs, get_params, generate_dataset
from ml_project.train import train_pipeline
from ml_project.predict import prediction_pipeline


def test_get_configs(config_path,
                     data_path, 
                     data_filename, 
                     train_filename, 
                     test_filename, 
                     target_col,
                     feature_cols):
    
    configs = get_configs(config_path)
    
    assert data_path == configs['input_data_path']
    assert data_filename == configs['data_file_name']
    assert target_col == configs['target_col']
    assert feature_cols == configs['feature_cols']
    
    assert get_params(configs['splitting_params'], 'val_size') > 0
    assert get_params(configs['splitting_params'], 'val_size') < 1
    

def test_dataset(data_path,
                 data_filename,
                 target_col,
                 feature_cols):
    
    data = pd.read_csv(data_path + data_filename)
    
    assert len(data) > 10
    assert target_col[0] in data.columns
    assert np.all(x.isdigit() for x in data[target_col])
    
    for col in feature_cols:
        assert col in data.columns
        assert np.all(x.isnumeric() for x in data[col])


def test_split_data(data_path, 
                    train_filename, 
                    test_filename):
    
    train_data = pd.read_csv(data_path + train_filename)
    test_data = pd.read_csv(data_path + test_filename)
    
    assert len(train_data) > 10
    assert len(test_data) > 10
    assert len(train_data) + len(test_data) > len(train_data)
    assert len(train_data) + len(test_data) > len(test_data)
    

def test_train(config_path,
                        test_data_path,
                        test_model_path):
    
    test_configs = get_configs(config_path)
    test_configs['input_data_path'] = test_data_path
    test_configs['output_model_path'] = test_model_path
    
    generate_dataset(test_configs, df_name='train.csv')
    train_pipeline(test_configs)
    
    assert os.path.exists(test_data_path)
    assert os.path.exists(test_model_path)

    
def test_predict(config_path,
                test_data_path,
                test_model_path,
                test_predictions_path):
    
    test_configs = get_configs(config_path)
    test_configs['input_data_path'] = test_data_path
    test_configs['output_model_path'] = test_model_path
    test_configs['output_data_path'] = test_predictions_path
    
    generate_dataset(test_configs, df_name='test.csv')
    prediction_pipeline(test_configs)
    
    assert os.path.exists(test_predictions_path)
    
    