from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount


with DAG('predict', start_date=days_ago(5),
         schedule_interval='@daily', catchup=False) as dag:

    data_sensor = FileSensor(
        task_id='data_sensor',
        poke_interval=20,
        timeout=60,
        retries=5,
        fs_conn_id='fs_connection',
        filepath="data/raw/{{ ds }}/data.csv",
    )

    pipeline_sensor = FileSensor(
        task_id='pipeline_sensor',
        poke_interval=20,
        timeout=60,
        retries=5,
        fs_conn_id='fs_connection',
        filepath="data/models/{{ ds }}/preprocessing_pipeline.gz",
    )

    predict = DockerOperator(
        task_id="docker-airflow-predict",
        image="airflow-predict",
        command="--input-dir /data/raw/{{ ds }} "
                "--output-dir /data/predictions/{{ ds }} "
                "--model-dir /data/models/{{ ds }}",
        do_xcom_push=True,
        mount_tmp_dir=True,
        mounts=[
            Mount(source="/home/alexander/Documents/MADE/Sem2/ML_prod/HW1_v2/airflow_ml_dags/data/", target="/data",
                  type='bind')]
    )

    [data_sensor, pipeline_sensor] >> predict