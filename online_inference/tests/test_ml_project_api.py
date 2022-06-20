from fastapi.testclient import TestClient

from main.ml_project_api import app, User, create_user


client = TestClient(app)


def test_make_predictions():
    test_user_id = -1
    test_user = User(age=50,
                     sex=0,
                     cp=3,
                     trestbps=150,
                     chol=250)

    create_user(test_user_id, test_user)

    predict_user_request = client.get(''.join(['predict/', str(test_user_id)]))
    assert predict_user_request.status_code == 200
    assert predict_user_request.text == '["Predicted condition is positive"]'

    test_user_id = -2
    test_user = User(age=30,
                     sex=1,
                     cp=0,
                     trestbps=100,
                     chol=100)

    create_user(test_user_id, test_user)

    predict_user_request = client.get(''.join(['predict/', str(test_user_id)]))
    assert predict_user_request.status_code == 200
    assert predict_user_request.text == '["Predicted condition is negative"]'
