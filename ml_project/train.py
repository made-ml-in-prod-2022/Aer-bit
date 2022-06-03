import sys
import pickle
import hydra
import logging
import xgboost

from ml_project.utils import preprocessing_pipeline


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@hydra.main(config_path='configs', config_name='config2.yaml')
def train_pipeline(configs):
    
    # Data preprocessing pipeline
    logger.info('Downloading training data from: {}'.format(hydra.utils.to_absolute_path(configs.train_data_path)))
    X_train, y_train = preprocessing_pipeline(hydra.utils.to_absolute_path(configs.train_data_path), 
                                              configs.feature_cols,
                                              configs.target_col)
    logger.info('Train data shape: {}'.format(X_train.shape))
    
    # Train model
    logger.info('Configure training parameters...')
    xgb_clf = xgboost.XGBClassifier(n_estimators=configs.train_params.n_estimators, 
                                    learning_rate=configs.train_params.lr,
                                    max_depth=configs.train_params.max_depth,
                                    random_state=configs.train_params.random_state)
    
    logger.info('Number of estimators: {}, learning rate: {}, max_depth: {}'.format(configs.train_params.n_estimators,
                                                                                    configs.train_params.lr,
                                                                                    configs.train_params.max_depth))
    logger.info('Start training...')
    xgb_clf.fit(X_train, y_train)
    logger.info('Finished training! Train accuracy score: {}'.format(xgb_clf.score(X_train, y_train)))
    
    # Save model
    pickle.dump(xgb_clf, open(hydra.utils.to_absolute_path(configs.output_model_path), "wb"))
        
        
if __name__ == '__main__':
    sys.exit(train_pipeline())