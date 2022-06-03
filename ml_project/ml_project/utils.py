import numpy as np
import pandas as pd
from Typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def preprocessing_pipeline(input_data_path: str,
                           feature_columns: list,
                           target_column: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    data = pd.read_csv(input_data_path)
    X, y = data[feature_columns], data[target_column]
    pipe = Pipeline(steps=[('impute', SimpleImputer())])
    X = pipe.fit_transform(X)
    return X, y
        
    
def generate_dataset(generated_train_path: str,
                     generated_test_path: str,
                     feature_columns: list,
                     target_column: list) -> None:
    
    # Generate train and test data
    df_train_size = np.random.randint(10, 10000)
    df_test_size = int(df_train_size * 0.5)
    df_columns = feature_columns + target_column
    n_cols = len(feature_columns)
    
    target_train_data = np.random.randint(2, size=df_train_size).reshape(-1, 1)
    features_train_data = np.random.rand(df_train_size, n_cols) * np.random.randint(-1e4, 1e4)
    
    target_test_data = np.random.randint(2, size=df_test_size).reshape(-1, 1)
    features_test_data = np.random.rand(df_test_size, n_cols) * np.random.randint(-1e4, 1e4)
    
    df_train = pd.DataFrame(np.hstack([features_train_data, target_train_data]), columns=df_columns)
    df_test = pd.DataFrame(np.hstack([features_test_data, target_test_data]), columns=df_columns)
    
    df_train.to_csv(generated_train_path)
    df_test.to_csv(generated_test_path)
