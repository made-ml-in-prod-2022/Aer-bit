import os
import sys
import yaml
import pickle
import logging
import numpy as np
import pandas as pd
import xgboost
from argparse import ArgumentParser
from ml_project.entities.pipeline_parameters import get_pipeline_parameters
from ml_project.utils import preprocessing_pipeline


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--conf_path', '-cf', help='Path to config file', default=None)
    return parser.parse_args()


def train_pipeline(parameters):
    
    
    # Data preprocessing pipeline
    logger.info('Downloading training data from: {}'.format(parameters.train_data_path))
    X_train, y_train = preprocessing_pipeline(parameters.train_data_path, 
                                              parameters.feature_cols,
                                              parameters.target_col)
    logger.info('Train data shape: {}'.format(X_train.shape))
    
    
    # Train model
    logger.info('Configure training parameters...')
    xgb_clf = xgboost.XGBClassifier(n_estimators=parameters.train_params.n_estimators, 
                                    learning_rate=parameters.train_params.lr,
                                    max_depth=parameters.train_params.max_depth,
                                    random_state=parameters.train_params.random_state)
    
    
    logger.info('Number of estimators: {}, learning rate: {}, max_depth: {}'.format(parameters.train_params.n_estimators,
                                                                                    parameters.train_params.lr,
                                                                                    parameters.train_params.max_depth))
    logger.info('Start training...')
    xgb_clf.fit(X_train, y_train)
    logger.info('Finished training! Train accuracy score: {}'.format(xgb_clf.score(X_train, y_train)))
    
    
    # Save model
    pickle.dump(xgb_clf, open(parameters.output_model_path, "wb"))
        
        
if __name__ == '__main__':
    args = parse_arguments()
    sys.exit(train_pipeline(get_pipeline_parameters(args.conf_path)))