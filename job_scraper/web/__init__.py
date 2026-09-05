"""Web triage UI: a local Flask app for browsing scored, relevant postings.

Run with: python -m job_scraper.web [--db-path data/jobs.db] [--host 127.0.0.1] [--port 5000]
"""
from __future__ import annotations

from pathlib import Path

from flask import Flask, abort, render_template, request

from job_scraper.db.repository import (
    get_posting_detail,
    get_relevant_postings,
    update_application_status,
)
from job_scraper.db.schema import init_db

VALID_APPLICATION_STATUSES = {"new", "applied", "skip"}


def _find_next_id(postings, after_id: int) -> int | None:
    ids = [p.id for p in postings]
    try:
        index = ids.index(after_id)
    except ValueError:
        return None
    return ids[index + 1] if index + 1 < len(ids) else None


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
        next_id = request.args.get("next_id", type=int)
        return render_template("_posting_detail.html", posting=posting, next_id=next_id)

    @app.post("/postings/<int:posting_id>/status")
    def update_status(posting_id: int):
        status = request.form.get("status")
        if status not in VALID_APPLICATION_STATUSES:
            abort(400)
        posting = get_posting_detail(app.config["DB_PATH"], posting_id)
        if posting is None:
            abort(404)
        update_application_status(app.config["DB_PATH"], posting.url, status)

        if status != "skip":
            updated = get_posting_detail(app.config["DB_PATH"], posting_id)
            next_id = request.args.get("next_id", type=int)
            return render_template("_posting_detail.html", posting=updated, next_id=next_id)

        # Skip auto-advances to the next posting in the list the user was viewing,
        # and the card for the just-skipped posting is removed via an htmx OOB swap
        # rather than a full list reload.
        remove_card = f'<div id="card-{posting_id}" hx-swap-oob="delete"></div>'
        next_id = request.args.get("next_id", type=int)
        next_posting = (
            get_posting_detail(app.config["DB_PATH"], next_id) if next_id is not None else None
        )
        if next_posting is None:
            return render_template("_detail_empty.html") + remove_card

        remaining = get_relevant_postings(app.config["DB_PATH"])
        new_next_id = _find_next_id(remaining, next_id)
        detail_html = render_template(
            "_posting_detail.html", posting=next_posting, next_id=new_next_id
        )
        return detail_html + remove_card

    return app
