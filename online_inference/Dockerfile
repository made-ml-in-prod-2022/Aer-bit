FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./main /code/main
COPY ./models /code/models
COPY ./configs /code/configs

CMD ["uvicorn", "main.ml_project_api:app", "--host", "0.0.0.0", "--port", "8000"]
