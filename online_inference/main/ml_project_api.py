import os
import pickle
import pandas as pd
from fastapi import FastAPI, Path, HTTPException
from typing import Optional
from pydantic import BaseModel, conint
import hydra
from hydra import initialize, compose
from sklearn.utils.validation import check_is_fitted


app = FastAPI()


class User(BaseModel):
    age: Optional[conint(ge=20, lt=120)] = 50
    sex: conint(ge=0, le=1)
    cp: Optional[conint(ge=0, le=3)] = 2
    trestbps: Optional[int] = 130
    chol: Optional[int] = 250
    fbs: Optional[int] = 0
    restecg: Optional[conint(ge=0, le=2)] = 1
    thalach: Optional[int] = 150
    exang: Optional[conint(ge=0, le=1)] = 0
    oldpeak: Optional[int] = 1
    slope: Optional[conint(ge=0, le=2)] = 0
    ca: Optional[int] = 0
    thal: Optional[int] = 1


users = {}


with initialize(config_path='../configs'):
    configs = compose(config_name='config.yaml', overrides=['hydra.run.dir=..'])
    model = pickle.load(open(hydra.utils.to_absolute_path(configs.output_model_path), "rb"))


@app.get('/')
def home() -> set:
    return {'Hello! You have accessed ML project API. To see list of available commands add /docs in the url.'}


@app.get('/health')
def health_check() -> None:
    assert os.path.exists(configs.output_model_path)
    check_is_fitted(model)

    raise HTTPException(status_code=200, detail='Ready to generate predictions!')


@app.post('/create-user/{user_id}')
def create_user(user_id: int, user: User) -> User:
    users[user_id] = user
    return users[user_id]


@app.get('/get-user/{user_id}')
def get_user(user_id: int = Path(None, description='The id of the user to inspect.')) -> User:
    if user_id not in users:
        raise HTTPException(status_code=400, detail='User with this ID does not exist')
    return users[user_id]


@app.get('/predict/{user_id}')
def make_predictions(user_id: int = Path(None, description='The ID of the user you would like to generate predictions for.')) -> set:
    if user_id not in users:
        raise HTTPException(status_code=400, detail='User with this ID does not exist')

    instance = pd.DataFrame()
    for feature in configs.feature_cols:
        instance.loc[0, feature] = getattr(users[user_id], feature)
    prediction = model.predict(instance.to_numpy())
    if prediction[0] == 0:
        return {'Predicted condition is negative.'}
    else:
        return {'Predicted condition is positive.'}
