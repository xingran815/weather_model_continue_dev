from airflow import DAG
from airflow.providers.http.operators.http import HttpOperator
import datetime



with DAG(
    dag_id='weather_pipeline_dag',
    description='weather pipeline for data substracting, \
        preprocessing, training, and prediction.',
    tags=['weather', 'pipeline'],
    schedule='*/10 * * * *',
    default_args={
        'owner': 'airflow',
        'start_date': datetime.datetime(2026, 1, 31),
    },
    catchup=False
) as my_dag:

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

    task_make_dataset >> task_preprocessing >> task_training