from __future__ import annotations

import argparse
import os

from job_scraper.web import create_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default=os.environ.get("JOB_SCRAPER_DB_PATH", "data/jobs.db"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app(db_path=args.db_path)
    app.run(host=args.host, port=args.port, debug=args.debug)
