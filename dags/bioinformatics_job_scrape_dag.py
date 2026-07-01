import os
from datetime import datetime

from airflow.decorators import dag, task

CONFIG_DIR = os.environ.get("JOB_SCRAPER_CONFIG_DIR", "/opt/airflow/config")
DB_PATH = os.environ.get("JOB_SCRAPER_DB_PATH", "/opt/airflow/data/jobs.db")


@dag(
    dag_id="bioinformatics_job_scrape",
    schedule="@daily",
    start_date=datetime(2026, 6, 1),
    catchup=False,
    tags=["job-search"],
)
def bioinformatics_job_scrape_dag():
    @task
    def scrape_greenhouse_task():
        from job_scraper.pipeline import run_source

        stats = run_source("greenhouse", CONFIG_DIR, DB_PATH)
        print(f"greenhouse: {stats}")

    @task
    def scrape_jobspy_task():
        from job_scraper.pipeline import run_source

        stats = run_source("jobspy", CONFIG_DIR, DB_PATH)
        print(f"jobspy: {stats}")

    scrape_greenhouse_task()
    scrape_jobspy_task()


bioinformatics_job_scrape_dag()
