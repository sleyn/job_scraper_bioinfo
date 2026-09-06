"""Web triage UI: a local Flask app for browsing scored, relevant postings.

Run with: python -m job_scraper.web [--db-path data/jobs.db] [--host 127.0.0.1] [--port 5000]
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

from flask import Flask, abort, render_template, request

from job_scraper.db.repository import (
    get_filter_options,
    get_posting_detail,
    get_relevant_postings,
    update_application_status,
)
from job_scraper.db.schema import init_db

VALID_APPLICATION_STATUSES = {"new", "applied", "skip"}
VALID_STATUS_FILTERS = {"new", "applied", "skip", "all"}
VALID_REMOTE_FILTERS = {"remote", "onsite"}


def _find_next_id(postings, after_id: int) -> int | None:
    ids = [p.id for p in postings]
    try:
        index = ids.index(after_id)
    except ValueError:
        return None
    return ids[index + 1] if index + 1 < len(ids) else None


def _filters_from_request(args) -> dict:
    """Parses the triage list's filter query params, shared by the list route and any
    caller that needs to know whether a filter is currently active (for empty-state
    messaging)."""
    status = args.get("status") or None
    if status is not None and status not in VALID_STATUS_FILTERS:
        abort(400)
    remote = args.get("remote") or None
    if remote is not None and remote not in VALID_REMOTE_FILTERS:
        abort(400)
    return {
        "include_all": args.get("show_all") == "1",
        "status": status,
        "companies": args.getlist("company") or None,
        "sources": args.getlist("source") or None,
        "min_score": args.get("min_score", type=float),
        "max_score": args.get("max_score", type=float),
        "keyword": args.get("q") or None,
        "remote": remote,
    }


def _filter_params(filters: dict) -> list[tuple[str, str]]:
    """The filter dict as (query-param-name, value) pairs, for rebuilding a URL that
    carries the active filters forward — used so the card-click and skip-auto-advance
    htmx requests keep operating on the same filtered list the user is looking at,
    without touching the browser's address bar (only the top-level `GET /` route push
    filters into the visible URL)."""
    params = []
    if filters["include_all"]:
        params.append(("show_all", "1"))
    if filters["status"]:
        params.append(("status", filters["status"]))
    for company in filters["companies"] or []:
        params.append(("company", company))
    for source in filters["sources"] or []:
        params.append(("source", source))
    if filters["min_score"] is not None:
        params.append(("min_score", filters["min_score"]))
    if filters["max_score"] is not None:
        params.append(("max_score", filters["max_score"]))
    if filters["keyword"]:
        params.append(("q", filters["keyword"]))
    if filters["remote"]:
        params.append(("remote", filters["remote"]))
    return params


def _url_builder(filters: dict):
    """Returns a `build(base_path, next_id=None)` closure over the active filters, for
    templates to construct htmx request URLs that carry the filters (and optionally
    next_id) forward as query params."""

    def build(base_path: str, next_id: int | None = None) -> str:
        params = list(_filter_params(filters))
        if next_id is not None:
            params.append(("next_id", next_id))
        query = urlencode(params)
        return f"{base_path}?{query}" if query else base_path

    return build


def _filters_active(filters: dict) -> bool:
    return any(
        [
            filters["include_all"],
            filters["status"] is not None,
            filters["companies"],
            filters["sources"],
            filters["min_score"] is not None,
            filters["max_score"] is not None,
            filters["keyword"],
            filters["remote"],
        ]
    )


def create_app(db_path: str | Path = "data/jobs.db") -> Flask:
    app = Flask(__name__)
    app.config["DB_PATH"] = str(db_path)
    init_db(app.config["DB_PATH"])

    @app.get("/")
    def triage_list():
        filters = _filters_from_request(request.args)
        postings = get_relevant_postings(app.config["DB_PATH"], **filters)
        filter_options = get_filter_options(app.config["DB_PATH"])
        return render_template(
            "triage_list.html",
            postings=postings,
            filter_options=filter_options,
            filters=filters,
            filters_active=_filters_active(filters),
            detail_url=_url_builder(filters),
        )

    @app.get("/postings/<int:posting_id>/detail")
    def posting_detail(posting_id: int):
        posting = get_posting_detail(app.config["DB_PATH"], posting_id)
        if posting is None:
            abort(404)
        filters = _filters_from_request(request.args)
        next_id = request.args.get("next_id", type=int)
        return render_template(
            "_posting_detail.html",
            posting=posting,
            next_id=next_id,
            status_url=_url_builder(filters),
        )

    @app.post("/postings/<int:posting_id>/status")
    def update_status(posting_id: int):
        status = request.form.get("status")
        if status not in VALID_APPLICATION_STATUSES:
            abort(400)
        posting = get_posting_detail(app.config["DB_PATH"], posting_id)
        if posting is None:
            abort(404)
        update_application_status(app.config["DB_PATH"], posting.url, status)
        filters = _filters_from_request(request.args)
        status_url = _url_builder(filters)

        if status != "skip":
            updated = get_posting_detail(app.config["DB_PATH"], posting_id)
            next_id = request.args.get("next_id", type=int)
            return render_template(
                "_posting_detail.html", posting=updated, next_id=next_id, status_url=status_url
            )

        # Skip auto-advances to the next posting in the same filtered list the user was
        # viewing (not an unfiltered scan of the whole table), and the card for the
        # just-skipped posting is removed via an htmx OOB swap rather than a full list
        # reload.
        remove_card = f'<div id="card-{posting_id}" hx-swap-oob="delete"></div>'
        next_id = request.args.get("next_id", type=int)
        next_posting = (
            get_posting_detail(app.config["DB_PATH"], next_id) if next_id is not None else None
        )
        if next_posting is None:
            return render_template("_detail_empty.html") + remove_card

        remaining = get_relevant_postings(app.config["DB_PATH"], **filters)
        new_next_id = _find_next_id(remaining, next_id)
        detail_html = render_template(
            "_posting_detail.html",
            posting=next_posting,
            next_id=new_next_id,
            status_url=status_url,
        )
        return detail_html + remove_card

    return app
