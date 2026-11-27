"""CLI entrypoint for the local web demo: python -m validator.web."""

from __future__ import annotations

import uvicorn
from dotenv import load_dotenv

from validator.web.app import create_app


def main() -> None:
    """Load environment variables and start the local server on 127.0.0.1:8000."""
    load_dotenv()
    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
