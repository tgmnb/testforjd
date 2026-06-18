"""TQSDK credentials — loaded from environment variables or JSON config.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def get_credentials() -> dict[str, str]:
    """Return tqsdk login credentials.

    Priority:
    1. TQ_USERNAME / TQ_PASSWORD environment variables
    2. config/credentials.json file (in the same dir as this module)
    """
    username = os.environ.get("TQ_USERNAME")
    password = os.environ.get("TQ_PASSWORD")

    if username and password:
        return {"username": username, "password": password}

    # Try JSON config file
    json_path = Path(__file__).resolve().parent / "credentials.json"
    if json_path.exists():
        try:
            with open(json_path) as f:
                data = json.load(f)
            if data.get("username") and data.get("password"):
                return data
        except Exception:
            pass

    raise RuntimeError(
        "TQSDK credentials not found. "
        "Set TQ_USERNAME and TQ_PASSWORD environment variables, "
        "or create a config/credentials.json file."
    )
