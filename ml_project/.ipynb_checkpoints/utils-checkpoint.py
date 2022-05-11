import yaml
import numpy as np
import pandas as pd


def get_configs(config_path):
    
    """Returns config parameters"""
    
    with open(config_path) as f:
        conf = yaml.safe_load(f.read())
        
    return conf


def get_params(parameters_lst, param_name):
    
    """
    Returns parameter value given list of parameter dictionaries
    
    parameters_lst: list of parameterrs
    param_name: name of parameter to return
    """
    
    for param in parameters_lst:
        if param_name in param:
            return param[param_name]
        
        
def generate_dataset(configs, df_name='train.csv'):
    
    """Generates dataset with random values"""
    
    df_size = np.random.randint(10, 10000)
    df_columns = configs['feature_cols'] + configs['target_col']
    n_cols = len(configs['feature_cols'])
    
    target_data = np.random.randint(2, size=df_size).reshape(-1, 1)
    features_data = np.random.rand(df_size, n_cols) * np.random.randint(-1e3, 1e3)
    
    df = pd.DataFrame(np.hstack([features_data, target_data]), columns=df_columns)
    df.to_csv(configs['input_data_path'] + df_name)

    
    
    