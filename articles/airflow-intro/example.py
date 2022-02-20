from datetime import datetime, timedelta

from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


@task
def my_task():
    print('Hello!')

with DAG(
    'tutorial_ctd',
    default_args=default_args,
    description='A simple tutorial DAG',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2021, 1, 9),
    catchup=False,
    tags=['example'],
    ) as dag:

    t1 = PythonOperator(
        task_id='print_hello',
        python_callable=my_task,
        dag=dag
    )
