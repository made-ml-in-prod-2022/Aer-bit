import os
import pytest
import numpy as np
import pandas as pd
from ml_project.entities.pipeline_parameters import get_pipeline_parameters 
from ml_project.utils import generate_dataset
from ml_project.train import train_pipeline
from ml_project.predict import prediction_pipeline


def test_get_pipeline_parameters(config_path,
                     data_path, 
                     data_filename, 
                     train_filename, 
                     test_filename, 
                     target_col,
                     feature_cols):
    
    parameters = get_pipeline_parameters(config_path)
    
    assert data_path == parameters.input_data_path
    assert data_filename == parameters.data_file_name
    assert target_col == parameters.target_col
    assert feature_cols == parameters.feature_cols
    
    assert parameters.splitting_params.val_size > 0
    assert parameters.splitting_params.val_size < 1
    

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
    

def test_train_pipeline(config_path,
                        test_data_path,
                        test_model_path):
    
    test_parameters = get_pipeline_parameters(config_path)
    test_parameters.input_data_path = test_data_path
    test_parameters.output_model_path = test_model_path
    
    generate_dataset(test_parameters, df_name='train.csv')
    train_pipeline(test_parameters)
    
    assert os.path.exists(test_data_path)
    assert os.path.exists(test_model_path)

    
def test_prediction_pipeline(config_path,
                test_data_path,
                test_model_path,
                test_predictions_path):
    
    test_parameters = get_pipeline_parameters(config_path)
    test_parameters.input_data_path = test_data_path
    test_parameters.output_model_path = test_model_path
    test_parameters.output_data_path = test_predictions_path
    
    generate_dataset(test_parameters, df_name='test.csv')
    prediction_pipeline(test_parameters)
    
    assert os.path.exists(test_predictions_path)
    
    