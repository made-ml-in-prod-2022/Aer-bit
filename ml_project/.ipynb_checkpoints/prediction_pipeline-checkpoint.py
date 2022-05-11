import os
import sys
import yaml
import pickle
import numpy as np
import pandas as pd
import utils

from argparse import ArgumentParser
from utils import get_configs, get_params, generate_dataset


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--conf_path', '-cf', help='Path to config file', default=None)
    return parser.parse_args()


def prediction_pipeline(parameters):

    # Get test data
    data_path = parameters['input_data_path']
    data = pd.read_csv(data_path + 'test.csv')
    
    # Prepare data 
    features = parameters['feature_cols']
    X_test = data[features]
    
    # Load model 
    model = pickle.load(open(parameters['output_model_path'], "rb"))
    
    # Make predictions
    predictions = model.predict(X_test)
    print('Predictions generated!')
    
    # Save predictions
    np.savetxt(data_path + 'predictions.csv', predictions, fmt='%s', delimiter=',')

    
if __name__ == '__main__':
    args = parse_arguments()
    parameters = get_configs(args.conf_path)
    sys.exit(prediction_pipeline(parameters))