from __future__ import annotations

import requests

USER_AGENT = "job-scraper/0.1 (personal job search tool)"


def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session
