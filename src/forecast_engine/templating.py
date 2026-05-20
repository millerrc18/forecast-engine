"""Shared Jinja2 templates instance — avoids circular imports."""

import json
from pathlib import Path
from fastapi.templating import Jinja2Templates

TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# Custom filters
def _from_json(value):
    """Parse a JSON string into a Python object; return [] on failure."""
    if not value:
        return []
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return []

templates.env.filters["from_json"] = _from_json
