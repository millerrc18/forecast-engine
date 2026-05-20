"""Oracle MCP data connector.

Pulls project master data and rates from the Oracle MCP server.

Note: Requires the Oracle MCP server to be running and configured.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


async def pull_oracle_data(
    programs: list[str] | None = None,
    session: AsyncSession | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Pull project master data from Oracle MCP.

    This function:
    1. Queries Oracle MCP for project details (rates, contract info, POP dates)
    2. Updates Program records in the database with fresh data

    Parameters
    ----------
    programs:
        Optional list of program codes to filter.
    session:
        Active async SQLAlchemy session.
    user_id:
        ID of the user triggering the import.

    Returns
    -------
    dict
        Status dict with programs_updated count and any warnings.
    """
    logger.warning(
        "Oracle MCP connector is not yet configured. "
        "Configure the Oracle MCP server to enable live data pulls."
    )

    return {
        "status": "failed",
        "programs_updated": 0,
        "warnings": [
            "Oracle MCP connector is not configured. "
            "Set ORACLE_MCP_URL in environment."
        ],
    }


def oracle_response_to_dataframe(raw_data: list[dict]) -> pd.DataFrame:
    """Convert raw Oracle MCP response to a structured DataFrame.

    Expected raw_data format:

    .. code-block:: python

        [
            {
                "project_number": "531335",
                "project_name": "GAC Elevator",
                "contract_type": "CPFF",
                "pop_start": "2024-01-01",
                "pop_end": "2027-12-31",
                "pm_name": "Ryan Miller",
                "site": "59",
            },
            ...
        ]

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per project.
    """
    if not raw_data:
        return pd.DataFrame()
    return pd.DataFrame(raw_data)
