import requests
import click


URL = 'http://127.0.0.1:8000/'


def predict_user(user_id: int) -> None:

    r = requests.get(''.join([URL, 'predict/', str(user_id)]))
    if r.status_code == 200:
        print(r.json()[0])
    else:
        print(r.json()['detail'])


@click.command(name='predict_user')
@click.argument('user_id')
def predict_user_command(user_id: int) -> None:
    predict_user(user_id)


if __name__ == '__main__':
    predict_user_command()
