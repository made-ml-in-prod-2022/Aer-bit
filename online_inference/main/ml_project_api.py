import os
import pickle
import pandas as pd
from fastapi import FastAPI, Path, HTTPException
from typing import Optional
from pydantic import BaseModel
import hydra
from hydra import initialize, compose
from sklearn.utils.validation import check_is_fitted


app = FastAPI()


class User(BaseModel):
    age: Optional[int] = 50
    sex: int
    cp: Optional[int] = 2
    trestbps: Optional[int] = 130
    chol: Optional[int] = 250
    fbs: Optional[int] = 0
    restecg: Optional[int] = 1
    thalach: Optional[int] = 150
    exang: Optional[int] = 0
    oldpeak: Optional[int] = 1
    slope: Optional[int] = 0
    ca: Optional[int] = 0
    thal: Optional[int] = 1


users = {}

@app.get('/')
def home():
    return {'Hello! You have accessed ML project API. To see list of available commands add /docs in the url.'}

@app.get('/get-user/{user_id}')
def get_user(user_id: int = Path(None, description='The id of the user to inspect.')):
    if user_id not in users:
        raise HTTPException(status_code=400, detail='User with this ID does not exist')
    return users[user_id]

@app.post('/create-user/{user_id}')
def create_user(user_id: int, user: User):
    if user_id in users:
        raise HTTPException(status_code=400, detail='User with this ID already exists')
    users[user_id] = user
    return users[user_id]

@app.get('/predict/{user_id}')
def make_predictions(user_id: int = Path(None, description='The ID of the user you would like to generate predictions for.'), config_name='config.yaml'):
    if user_id not in users:
        raise HTTPException(status_code=400, detail='User with this ID does not exist')

    with initialize(config_path='../configs'):
        configs = compose(config_name=config_name,  overrides=['hydra.run.dir=..'])
        model = pickle.load(open(hydra.utils.to_absolute_path(configs.output_model_path), "rb"))
        instance = pd.DataFrame()
        for feature in configs.feature_cols:
            instance.loc[0, feature] = getattr(users[user_id], feature)
        prediction = model.predict(instance.to_numpy())
        if prediction[0] == 0:
            return {'Predicted condition is negative'}
        else:
            return {'Predicted condition is positive'}

@app.get('/health/{config_name}')
def health_check(config_name: str = Path(None, description='Config file name you would like to use.')):
    with initialize(config_path='../configs'):
        configs = compose(config_name=config_name,  overrides=['hydra.run.dir=..'])

        assert os.path.exists(configs.output_model_path)

        model = pickle.load(open(hydra.utils.to_absolute_path(configs.output_model_path), "rb"))
        check_is_fitted(model)
            
    raise HTTPException(status_code=200, detail='Ready to generate predictions!')

