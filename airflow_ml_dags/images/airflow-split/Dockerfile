FROM python:3.6-slim-stretch

COPY requirements.txt /requirements.txt
RUN pip install --no-cache -r requirements.txt

COPY split.py /split.py

ENTRYPOINT ["python", "split.py"]