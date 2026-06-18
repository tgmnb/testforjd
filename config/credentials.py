"""TQSDK credentials — loaded from environment variables.

Set these in your shell:
    export TQ_USERNAME="your_username"
    export TQ_PASSWORD="your_password"
"""

from __future__ import annotations

import os


def get_credentials() -> dict[str, str]:
    """Return tqsdk login credentials from environment or config file."""
    username = os.environ.get("TQ_USERNAME")
    password = os.environ.get("TQ_PASSWORD")

    if username and password:
        return {"username": username, "password": password}

    raise RuntimeError(
        "TQSDK credentials not found. "
        "Set TQ_USERNAME and TQ_PASSWORD environment variables, "
        "or create a config/credentials.json file."
    )
