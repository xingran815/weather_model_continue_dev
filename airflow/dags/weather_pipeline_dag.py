from airflow.sdk import dag, task, Variable
from airflow.providers.http.operators.http import HttpOperator
from airflow.providers.http.sensors.http import HttpHook, HttpSensor
import datetime



DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": datetime.timedelta(seconds=60),
    "start_date": datetime.datetime(2026, 1, 31),
}


@dag(
    dag_id="weather_pipeline_dag",
    description=(
        "weather pipeline for data substracting, preprocessing, training, and"
        " prediction."
    ),
    tags=["weather", "pipeline"],
    schedule="*/10 * * * *",
    default_args=DEFAULT_ARGS,
    catchup=False,
)
def weather_pipeline_dag():

    check_model_service = HttpSensor(
        task_id='check_model_service',
        http_conn_id='model_api',
        endpoint='/',
        poke_interval=30,
        timeout=120,
    )


    @task(task_id='make_dataset')
    def task_make_dataset():
        duration = int(Variable.get('duration', default=2))
        if duration < 10:
            Variable.set('duration', str(duration+1))
        resp = HttpHook(method="GET", http_conn_id="model_api").run(
            f"/make_dataset?duration={duration}")
        resp.raise_for_status()


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
    task_make_dataset() >> \
    task_preprocessing >> \
    task_training


dag = weather_pipeline_dag()