import os
import sys
import yaml
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from ml_project.entities.pipeline_parameters import get_pipeline_parameters 


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--conf_path', '-cf', help='Path to config file', default=None)
    return parser.parse_args()


def split_data(parameters):
    
    # Load data
    data = pd.read_csv(parameters.input_data_path)
        
    # Train test split          
    train, test = train_test_split(data, 
                                   test_size=parameters.splitting_params.val_size, 
                                   random_state=parameters.splitting_params.random_state)
    
    # Save train and test data
    train.to_csv(parameters.train_data_path, index=None)
    test.to_csv(parameters.test_data_path, index=None)
              

if __name__ == '__main__':
    args = parse_arguments()
    sys.exit(split_data(get_pipeline_parameters(args.conf_path)))