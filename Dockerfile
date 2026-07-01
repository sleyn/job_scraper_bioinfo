FROM apache/airflow:2.9.3-python3.12

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

COPY pyproject.toml /opt/airflow/pyproject.toml
COPY job_scraper /opt/airflow/job_scraper
RUN pip install --no-cache-dir -e /opt/airflow
