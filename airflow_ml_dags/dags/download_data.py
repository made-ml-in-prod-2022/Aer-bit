from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

with DAG('docker_download', start_date=days_ago(5),
         schedule_interval='@daily', catchup=False) as dag:

    download = DockerOperator(
        task_id="docker-airflow-download",
        image="airflow-download",
        command="/data/raw/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=True,
        mount_tmp_dir=True,
        mounts=[Mount(source="/home/alexander/Documents/MADE/Sem2/ML_prod/HW1_v2/airflow_ml_dags/data/", target="/data", type='bind')]
    )


