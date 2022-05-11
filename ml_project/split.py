import os
import sys
import yaml
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from ml_project.utils import get_params, get_configs


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--conf_path', '-cf', help='Path to config file', default=None)
    return parser.parse_args()


def split_data(parameters):

    # Get split params
    split_params = parameters['splitting_params']
    test_size = get_params(split_params, 'val_size')
    random_state = get_params(split_params, 'random_state') 
    
    # Load data
    data_path = parameters['input_data_path']
    fname = parameters['data_file_name']
    data = pd.read_csv(data_path + fname)
    
    # Train test split          
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)
    
    # Save train and test data
    train.to_csv(data_path + 'train.csv', index=None)
    test.to_csv(data_path + 'test.csv', index=None)
              

if __name__ == '__main__':
    args = parse_arguments()
    parameters = get_configs(args.conf_path)
    sys.exit(split_data(parameters))