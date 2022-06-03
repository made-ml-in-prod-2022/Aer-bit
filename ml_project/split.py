import sys
import pandas as pd
import hydra
from sklearn.model_selection import train_test_split


@hydra.main(config_path='configs', config_name='config.yaml')
def split_data(configs):
    
    # Load data
    data = pd.read_csv(hydra.utils.to_absolute_path(configs.input_data_path))
        
    # Train test split          
    train, test = train_test_split(data, 
                                   test_size=configs.splitting_params.val_size, 
                                   random_state=configs.splitting_params.random_state)
    
    # Save train and test data
    train.to_csv(hydra.utils.to_absolute_path(configs.train_data_path), index=None)
    test.to_csv(hydra.utils.to_absolute_path(configs.test_data_path), index=None)
              

if __name__ == '__main__':
    sys.exit(split_data())
