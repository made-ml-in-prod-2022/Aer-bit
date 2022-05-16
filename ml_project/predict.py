import os
import sys
import yaml
import pickle
import hydra
from omegaconf import DictConfig
import logging
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@hydra.main(config_path='configs', config_name='config.yaml')
def prediction_pipeline(configs):
    #orig_cwd = hydra.utils.get_original_cwd()

    # Get test data
    data = pd.read_csv(hydra.utils.to_absolute_path(configs.test_data_path))
    logger.info('Loading test data from: {}'.format(hydra.utils.to_absolute_path(configs.test_data_path)))
    
    # Prepare data 
    X_test = data[configs.feature_cols]
    
    # Load model 
    model = pickle.load(open(hydra.utils.to_absolute_path(configs.output_model_path), "rb"))
    logger.info('Loading model from: {}'.format(hydra.utils.to_absolute_path(configs.output_model_path)))
    
    # Make predictions
    predictions = model.predict(X_test)
    logger.info('Generating predictions...')
    
    # Save predictions
    logger.info('Saving predictions to {}'.format(hydra.utils.to_absolute_path(configs.output_data_path)))
    np.savetxt(hydra.utils.to_absolute_path(configs.output_data_path), predictions, fmt='%s', delimiter=',')
    logger.info('Predictions generated!')

    
if __name__ == '__main__':
    sys.exit(prediction_pipeline())