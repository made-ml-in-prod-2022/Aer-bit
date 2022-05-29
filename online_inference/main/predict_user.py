import requests
import click


def predict_user(user_id):

    app_url = 'http://127.0.0.1:8000/'
    func = 'predict/'
    r = requests.get(''.join([app_url, func, str(user_id)]))

    print(r.text)


@click.command(name='predict_user')
@click.argument('user_id')
def predict_user_command(user_id: int):
    predict_user(user_id)


if __name__ == '__main__':
    predict_user_command()