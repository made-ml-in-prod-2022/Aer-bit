import requests
import click


URL = 'http://127.0.0.1:8000/'


def predict_user(user_id: int) -> None:

    if check_input(user_id):
        r = requests.get(''.join([URL, 'predict/', str(user_id)]))
        if r.status_code == 200:
            print(r.json()[0])
        else:
            print(r.json()['detail'])


def check_input(user_id: int) -> None:
    r = requests.get(''.join([URL, 'get-user/', str(user_id)]))
    if r.status_code == 200:
        user = r.json()
        for attr in user:
            assert user[attr] >= 0
        assert user['age'] <= 150
        assert user['sex'] in [0, 1]
        assert user['cp'] in [0, 1, 2, 3]
        assert user['restecg'] in [0, 1, 2]
        assert user['exang'] in [0, 1]
        assert user['slope'] in [0, 1, 2]
        return True
    else:
        print(r.json()['detail'])
        return False


@click.command(name='predict_user')
@click.argument('user_id')
def predict_user_command(user_id: int) -> None:
    predict_user(user_id)


if __name__ == '__main__':
    predict_user_command()
