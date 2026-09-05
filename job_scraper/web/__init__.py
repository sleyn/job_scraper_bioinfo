"""Web triage UI: a local Flask app for browsing scored, relevant postings.

Run with: python -m job_scraper.web [--db-path data/jobs.db] [--host 127.0.0.1] [--port 5000]
"""
from __future__ import annotations

from pathlib import Path

from flask import Flask, abort, render_template

from job_scraper.db.repository import get_posting_detail, get_relevant_postings
from job_scraper.db.schema import init_db


def create_app(db_path: str | Path = "data/jobs.db") -> Flask:
    app = Flask(__name__)
    app.config["DB_PATH"] = str(db_path)
    init_db(app.config["DB_PATH"])

    @app.get("/")
    def triage_list():
        postings = get_relevant_postings(app.config["DB_PATH"])
        return render_template("triage_list.html", postings=postings)

    @app.get("/postings/<int:posting_id>/detail")
    def posting_detail(posting_id: int):
        posting = get_posting_detail(app.config["DB_PATH"], posting_id)
        if posting is None:
            abort(404)
        return render_template("_posting_detail.html", posting=posting)

    return app
