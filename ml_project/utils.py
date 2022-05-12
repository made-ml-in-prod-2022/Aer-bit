import yaml
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def preprocessing_pipeline(input_data_path,
                           feature_columns,
                           target_column):
    
    data = pd.read_csv(input_data_path)
    X, y = data[feature_columns], data[target_column]
    pipe = Pipeline(steps=[('impute', SimpleImputer())])
    X = pipe.fit_transform(X)
    return X, y
        
    
def generate_dataset(configs, df_name='train.csv'):
    
    """Generates dataset with random values"""
    
    df_size = np.random.randint(10, 10000)
    df_columns = configs.feature_cols + configs.target_col
    n_cols = len(configs.feature_cols)
    
    target_data = np.random.randint(2, size=df_size).reshape(-1, 1)
    features_data = np.random.rand(df_size, n_cols) * np.random.randint(-1e3, 1e3)
    
    df = pd.DataFrame(np.hstack([features_data, target_data]), columns=df_columns)
    df.to_csv(configs.input_data_path + df_name)

    
    
    