import os
import sys
import yaml
import pickle
import logging
import numpy as np
import pandas as pd
import xgboost
from ml_project import entities

from argparse import ArgumentParser
from ml_project.entities.pipeline_parameters import get_pipeline_parameters 


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--conf_path', '-cf', help='Path to config file', default=None)
    return parser.parse_args()


def train_pipeline(parameters):
    
    # Get train data
    data_path = parameters.input_data_path + 'train.csv'
    data = pd.read_csv(data_path)
    logger.info('Downloading training data from: {}'.format(data_path))
    
    # Get train params
    logger.info('Configure training parameters...')
    n_estimators = parameters.train_params.n_estimators
    lr = parameters.train_params.lr
    max_depth = parameters.train_params.max_depth
    random_state = parameters.train_params.random_state
    logger.info('Number of estimators: {}, learning rate: {}, max_depth: {}'.format(n_estimators, lr, max_depth))
    
    # Prepare data for training
    features = parameters.feature_cols
    print(features)
    target = parameters.target_col
    X_train = data[features]
    y_train = data[target]
    logger.info('Train data shape: {}'.format(X_train.shape))
    
    # Train model
    logger.info('Start training...')
    xgb_clf = xgboost.XGBClassifier(n_estimators=n_estimators, learning_rate=lr, max_depth=max_depth, random_state=random_state)
    xgb_clf.fit(X_train, y_train)
    logger.info('Finished training! Train accuracy score: {}'.format(xgb_clf.score(X_train, y_train)))
    
    # Save model
    pickle.dump(xgb_clf, open(parameters.output_model_path, "wb"))
        
        
if __name__ == '__main__':
    args = parse_arguments()
    sys.exit(train_pipeline(get_pipeline_parameters(args.conf_path)))