FROM apache/airflow:2.9.3-python3.12

# The scoring stage needs torch, but torch's default wheels pull ~16 NVIDIA CUDA
# packages — several GB — that this container can never use: Docker on macOS passes no
# GPU through, and scoring runs on CPU. Installing the CPU-only build first means the
# editable install below finds the requirement already satisfied and skips them.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml /opt/airflow/pyproject.toml
COPY job_scraper /opt/airflow/job_scraper
RUN pip install --no-cache-dir -e "/opt/airflow[scoring]"
