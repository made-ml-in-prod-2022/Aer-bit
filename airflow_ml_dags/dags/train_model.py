from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount


with DAG('train_model', start_date=days_ago(5),
         schedule_interval='@weekly', catchup=False) as dag:

        data_sensor = FileSensor(
            task_id='data_sensor',
            poke_interval=20,
            timeout=60,
            retries=5,
            fs_conn_id='fs_connection',
            filepath="data/raw/{{ ds }}/data.csv",
        )

        target_sensor = FileSensor(
            task_id='target_sensor',
            poke_interval=20,
            timeout=60,
            retries=5,
            fs_conn_id='fs_connection',
            filepath="data/raw/{{ ds }}/target.csv",
        )

        preprocess = DockerOperator(
            task_id="docker-airflow-preprocess",
            image="airflow-preprocess",
            command="--input-dir /data/raw/{{ ds }} "
                    "--output-dir /data/processed/{{ ds }} "
                    "--model-dir /data/models/{{ ds }}",
            do_xcom_push=True,
            mount_tmp_dir=True,
            mounts=[
                Mount(source="/home/alexander/Documents/MADE/Sem2/ML_prod/HW1_v2/airflow_ml_dags/data/", target="/data",
                      type='bind')]
        )

        split = DockerOperator(
            task_id="docker-airflow-split",
            image="airflow-split",
            command="--input-dir /data/processed/{{ ds }} "
                    "--output-dir /data/processed/{{ ds }}",
            do_xcom_push=True,
            mount_tmp_dir=True,
            mounts=[
                Mount(source="/home/alexander/Documents/MADE/Sem2/ML_prod/HW1_v2/airflow_ml_dags/data/", target="/data",
                      type='bind')]
        )

        train = DockerOperator(
            task_id="docker-airflow-train",
            image="airflow-train",
            command="--input-dir /data/processed/{{ ds }} "
                    "--model-dir /data/models/{{ ds }}",
            do_xcom_push=True,
            mount_tmp_dir=True,
            mounts=[
                Mount(source="/home/alexander/Documents/MADE/Sem2/ML_prod/HW1_v2/airflow_ml_dags/data/", target="/data",
                      type='bind')]
        )

        validate = DockerOperator(
            task_id="docker-airflow-validate",
            image="airflow-validate",
            command="--input-dir /data/processed/{{ ds }} "
                    "--model-dir /data/models/{{ ds }}",
            do_xcom_push=True,
            mount_tmp_dir=True,
            mounts=[
                Mount(source="/home/alexander/Documents/MADE/Sem2/ML_prod/HW1_v2/airflow_ml_dags/data/", target="/data",
                      type='bind')]
        )

        [data_sensor, target_sensor] >> preprocess >> split >> train >> validate