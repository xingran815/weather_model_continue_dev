from airflow import DAG
from airflow.providers.http.operators.http import HttpOperator
from airflow.providers.http.sensors.http import HttpSensor
import datetime



with DAG(
    dag_id='weather_pipeline_dag',
    description='weather pipeline for data substracting, \
        preprocessing, training, and prediction.',
    tags=['weather', 'pipeline'],
    schedule='*/10 * * * *',
    default_args={
        'owner': 'airflow',
        'retries': 2,
        'retry_delay': datetime.timedelta(seconds=60),
        'start_date': datetime.datetime(2026, 1, 31),
    },
    catchup=False,
) as my_dag:


    check_model_service = HttpSensor(
        task_id='check_model_service',
        http_conn_id='model_api',
        endpoint='/',
        poke_interval=30,
        timeout=120,
    )

    task_make_dataset = HttpOperator(
        task_id="make_dataset",
        http_conn_id="model_api",
        endpoint="/make_dataset",
        method="GET",
        headers={},
    )

    task_preprocessing = HttpOperator(
        task_id="preprocessing",
        http_conn_id="model_api",
        endpoint="/preprocessing",
        method="GET",
        headers={},
    )

    task_training = HttpOperator(
        task_id="training",
        http_conn_id="model_api",
        endpoint="/training",
        method="GET",
        headers={},
    )

    check_model_service >> \
    task_make_dataset >> \
    task_preprocessing >> \
    task_training