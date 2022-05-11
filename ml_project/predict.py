import os
import sys
import yaml
import pickle
import logging
import numpy as np
import pandas as pd
from ml_project import utils

from argparse import ArgumentParser
from ml_project.utils import get_configs, get_params, generate_dataset


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--conf_path', '-cf', help='Path to config file', default=None)
    return parser.parse_args()


def prediction_pipeline(parameters):

    # Get test data
    data_path = parameters['input_data_path'] + 'test.csv'
    data = pd.read_csv(data_path)
    logger.info('Loading test data from: {}'.format(data_path))
    
    # Prepare data 
    features = parameters['feature_cols']
    X_test = data[features]
    
    # Load model 
    model = pickle.load(open(parameters['output_model_path'], "rb"))
    logger.info('Loading model from: {}'.format(parameters['output_model_path']))
    
    # Make predictions
    predictions = model.predict(X_test)
    print('Generating predictions...')
    
    # Save predictions
    np.savetxt(data_path + 'predictions.csv', predictions, fmt='%s', delimiter=',')
    logger.info('Predictions generated!')

    
if __name__ == '__main__':
    args = parse_arguments()
    parameters = get_configs(args.conf_path)
    sys.exit(prediction_pipeline(parameters))