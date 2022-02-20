from airflow.decorators import task

@task
def my_task():
    print('Hello!')
