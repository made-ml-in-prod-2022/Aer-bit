import os
import sys
import yaml
import pickle
import logging
import numpy as np
import pandas as pd

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


def prediction_pipeline(parameters):

    # Get test data
    data_path = parameters.input_data_path + 'test.csv'
    data = pd.read_csv(data_path)
    logger.info('Loading test data from: {}'.format(data_path))
    
    # Prepare data 
    features = parameters.feature_cols
    X_test = data[features]
    
    # Load model 
    model = pickle.load(open(parameters.output_model_path, "rb"))
    logger.info('Loading model from: {}'.format(parameters.output_model_path))
    
    # Make predictions
    predictions = model.predict(X_test)
    logger.info('Generating predictions...')
    
    # Save predictions
    logger.info('Saving predictions to {}'.format(parameters.output_data_path))
    np.savetxt(parameters.output_data_path, predictions, fmt='%s', delimiter=',')
    logger.info('Predictions generated!')

    
if __name__ == '__main__':
    args = parse_arguments()
    sys.exit(prediction_pipeline(get_pipeline_parameters(args.conf_path)))