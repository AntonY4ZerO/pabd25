from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "pabd25",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="zinyakov_daily_csv_parse_and_train",
    default_args=default_args,
    schedule="@daily",  # каждый день
    catchup=False,
    tags=["csv", "model"],
) as dag:

    parse_data = BashOperator(
        task_id="zin_parse_csv",
        bash_command="python ~/airflow/Zinyakov/src/zin_parse_cian.py -p 1",
    )

    train_model = BashOperator(
        task_id="zin_train_model",
        bash_command="python ~/airflow/Zinyakov/src/zin_train_model.py -m xgboost",
    )

    parse_data >> train_model
