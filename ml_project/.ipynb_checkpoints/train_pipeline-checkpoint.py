import os
import sys
import yaml
import pickle
import numpy as np
import pandas as pd
import xgboost

from argparse import ArgumentParser
from utils import get_configs, get_params, generate_dataset


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--conf_path', '-cf', help='Path to config file', default=None)
    return parser.parse_args()


def train_pipeline(parameters):

    # Get train data
    data_path = parameters['input_data_path']
    data = pd.read_csv(data_path + 'train.csv')
    
    # Get train params
    train_params = parameters['train_params']
    n_estimators = get_params(train_params, 'n_estimators')
    lr = get_params(train_params, 'lr')
    max_depth = get_params(train_params, 'max_depth')
    random_state = get_params(train_params, 'random_state')
    
    # Prepare data for training
    features = parameters['feature_cols']
    target = parameters['target_col']
    X_train = data[features]
    y_train = data[target]
    
    # Train model
    xgb_clf = xgboost.XGBClassifier(n_estimators=n_estimators, learning_rate=lr, max_depth=max_depth, random_state=random_state)
    xgb_clf.fit(X_train, y_train)
    print('Finished training! Train accuracy score: {}'.format(xgb_clf.score(X_train, y_train)))
    
    # Save model
    pickle.dump(xgb_clf, open(parameters['output_model_path'], "wb"))
        
        
if __name__ == '__main__':
    args = parse_arguments()
    parameters = get_configs(args.conf_path)
    sys.exit(train_pipeline(parameters))