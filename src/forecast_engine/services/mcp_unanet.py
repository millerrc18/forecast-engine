"""Unanet MCP data connector.

Pulls labor hours data from the Unanet MCP server and converts it to the
standard DataFrame format used by the importer service.

Note: Requires the Unanet MCP server to be running and configured.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from forecast_engine.services.importer import import_file

logger = logging.getLogger(__name__)


async def pull_unanet_hours(
    start_date: date,
    end_date: date,
    programs: list[str] | None = None,
    session: AsyncSession | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Pull labor hours from Unanet MCP and import into the database.

    This function:
    1. Queries the Unanet MCP server for hours by program/person/cost pool/period
    2. Converts the response to a CSV-compatible DataFrame
    3. Writes it to a temp file
    4. Calls import_file() to upsert into the database

    Parameters
    ----------
    start_date:
        First day of the date range to pull.
    end_date:
        Last day of the date range to pull.
    programs:
        Optional list of program codes to filter (default: all).
    session:
        Active async SQLAlchemy session.
    user_id:
        ID of the user triggering the import.

    Returns
    -------
    dict
        Import result from import_file().
    """
    # TODO: Replace with actual MCP call when Unanet MCP server is configured.
    # The MCP call would look something like:
    #   result = await mcp_client.call("unanet_pull_hours", {
    #       "start_date": start_date.isoformat(),
    #       "end_date": end_date.isoformat(),
    #       "programs": programs,
    #   })

    logger.warning(
        "Unanet MCP connector is not yet configured. "
        "Use CSV upload or configure the Unanet MCP server."
    )

    return {
        "import_id": None,
        "status": "failed",
        "rows_imported": 0,
        "rows_skipped": 0,
        "periods": [],
        "programs": [],
        "warnings": [
            "Unanet MCP connector is not configured. "
            "Set UNANET_MCP_URL in environment or use CSV upload."
        ],
    }


def unanet_response_to_dataframe(raw_data: list[dict]) -> pd.DataFrame:
    """Convert raw Unanet MCP response to standard importer DataFrame format.

    Expected raw_data format (from Unanet MCP):

    .. code-block:: python

        [
            {
                "project_code": "531335",
                "person_name": "Smith, John",
                "cost_pool": "BAMTL",
                "period": "MAY-26",
                "hours": 160.5,
                "rate": 48.50,
                "task_code": "PROD",
            },
            ...
        ]

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: program, period, cost_pool, person, hours,
        rate, activity_type.
    """
    if not raw_data:
        return pd.DataFrame()

    df = pd.DataFrame(raw_data)

    # Map MCP field names to importer expected names
    column_map = {
        "project_code": "program",
        "person_name": "person",
        "cost_pool": "cost_pool",
        "period": "period",
        "hours": "hours",
        "rate": "rate",
        "task_code": "activity_type",
    }

    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
    return df
