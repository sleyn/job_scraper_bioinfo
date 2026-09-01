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
    def scrape_lever_task():
        from job_scraper.pipeline import run_source

        stats = run_source("lever", CONFIG_DIR, DB_PATH)
        print(f"lever: {stats}")

    @task
    def scrape_ashby_task():
        from job_scraper.pipeline import run_source

        stats = run_source("ashby", CONFIG_DIR, DB_PATH)
        print(f"ashby: {stats}")

    @task
    def scrape_workday_task():
        from job_scraper.pipeline import run_source

        stats = run_source("workday", CONFIG_DIR, DB_PATH)
        print(f"workday: {stats}")

    @task
    def scrape_jobspy_task():
        from job_scraper.pipeline import run_source

        stats = run_source("jobspy", CONFIG_DIR, DB_PATH)
        print(f"jobspy: {stats}")

    @task
    def score_postings_task():
        from job_scraper.pipeline import score_pending_postings

        stats = score_pending_postings(CONFIG_DIR, DB_PATH)
        print(f"scoring: {stats}")

    gh = scrape_greenhouse_task()
    lv = scrape_lever_task()
    ab = scrape_ashby_task()
    wd = scrape_workday_task()
    js = scrape_jobspy_task()
    sc = score_postings_task()
    [gh, lv, ab, wd, js] >> sc


bioinformatics_job_scrape_dag()
