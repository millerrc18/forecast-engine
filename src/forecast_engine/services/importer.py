"""CSV / Excel importer service for labor hours data.

Supports Unanet pivot-table exports and Oracle CSV extracts. Parses raw files
into normalized DataFrames, maps columns to the schema, upserts ActualHours
rows, and writes a DataImport audit record.

Public API
----------
import_file(file_path, source, user_id, session) -> dict
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from forecast_engine.models.actuals import (
    ACTIVITY_TYPES,
    COST_POOLS,
    ActualHours,
    DataImport,
)
from forecast_engine.models.program import ForecastPeriod, Program

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column alias map — each canonical name maps to the raw header variants we
# expect to encounter across Unanet and Oracle exports.
# ---------------------------------------------------------------------------

COLUMN_ALIASES: dict[str, list[str]] = {
    "program": ["program", "project", "charge_no", "charge_#", "project_code"],
    "period": ["period", "month", "fiscal_period"],
    "cost_pool": ["cost_pool", "costpool", "labor_category", "cost_element"],
    "person": ["person", "employee", "name", "resource"],
    "hours": ["hours", "total_hours", "hrs", "quantity"],
    "rate": ["rate", "hourly_rate", "bill_rate"],
    "activity_type": ["activity_type", "task", "task_code", "activity"],
}

# Cost-pool synonyms that appear in raw exports before normalization.
COST_POOL_SYNONYMS: dict[str, str] = {
    "ba": "BAMTL",
    "bamtl": "BAMTL",
    "bam": "BAMTL",
    "eng": "ENGTL",
    "engtl": "ENGTL",
    "engineering": "ENGTL",
    "mfg": "MFGTL",
    "mfgtl": "MFGTL",
    "manufacturing": "MFGTL",
}

# Month abbreviation → number used for normalizing period labels to "MMM-YY".
_MONTH_ABBREVS = {
    "jan": "JAN", "feb": "FEB", "mar": "MAR", "apr": "APR",
    "may": "MAY", "jun": "JUN", "jul": "JUL", "aug": "AUG",
    "sep": "SEP", "oct": "OCT", "nov": "NOV", "dec": "DEC",
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _normalize_header(raw: str) -> str:
    """Lowercase, strip whitespace, replace spaces/dashes with underscores."""
    return re.sub(r"[\s\-]+", "_", str(raw).strip().lower())


def _resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    """Return a mapping of {canonical_name: actual_df_column} for recognised columns.

    Walks every column in the DataFrame and matches it against COLUMN_ALIASES.
    The *first* match wins when multiple raw columns map to the same canonical name.
    """
    normalized_cols = {_normalize_header(c): c for c in df.columns}
    resolved: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias_norm = _normalize_header(alias)
            if alias_norm in normalized_cols and canonical not in resolved:
                resolved[canonical] = normalized_cols[alias_norm]
                break
    return resolved


def _normalize_period_label(raw: str) -> str | None:
    """Coerce various period string formats to "MMM-YY" (e.g. "JAN-25").

    Handles:
    - "JAN-25", "jan-25"           → "JAN-25"
    - "Jan 2025", "January 2025"   → "JAN-25"
    - "2025-01", "01/2025"         → "JAN-25"
    - "2501"                       → "JAN-25"
    """
    raw = str(raw).strip()

    # Already "MMM-YY" or "MMM-YYYY"
    m = re.match(r"^([A-Za-z]{3})-(\d{2,4})$", raw)
    if m:
        abbrev = m.group(1).lower()
        year = m.group(2)[-2:]
        if abbrev in _MONTH_ABBREVS:
            return f"{_MONTH_ABBREVS[abbrev]}-{year}"

    # "Month YYYY" — e.g. "January 2025" or "Jan 2025"
    m = re.match(r"^([A-Za-z]+)\s+(\d{4})$", raw)
    if m:
        abbrev = m.group(1)[:3].lower()
        year = m.group(2)[-2:]
        if abbrev in _MONTH_ABBREVS:
            return f"{_MONTH_ABBREVS[abbrev]}-{year}"

    # "YYYY-MM" or "MM/YYYY"
    m = re.match(r"^(\d{4})-(\d{2})$", raw)
    if m:
        month_num = int(m.group(2))
        year = m.group(1)[-2:]
        abbrev = list(_MONTH_ABBREVS.values())[month_num - 1]
        return f"{abbrev}-{year}"

    m = re.match(r"^(\d{2})/(\d{4})$", raw)
    if m:
        month_num = int(m.group(1))
        year = m.group(2)[-2:]
        abbrev = list(_MONTH_ABBREVS.values())[month_num - 1]
        return f"{abbrev}-{year}"

    # "YYMM" compact form (e.g. "2501" → JAN-25)
    m = re.match(r"^(\d{2})(\d{2})$", raw)
    if m:
        year = m.group(1)
        month_num = int(m.group(2))
        if 1 <= month_num <= 12:
            abbrev = list(_MONTH_ABBREVS.values())[month_num - 1]
            return f"{abbrev}-{year}"

    return None


def _normalize_cost_pool(raw: str) -> str | None:
    """Map raw cost-pool string to a canonical COST_POOLS value or None."""
    key = str(raw).strip().lower()
    if key in COST_POOL_SYNONYMS:
        return COST_POOL_SYNONYMS[key]
    upper = key.upper()
    if upper in COST_POOLS:
        return upper
    return None


def _normalize_activity_type(raw: str | None) -> str:
    """Map raw activity-type string to a canonical ACTIVITY_TYPES value.

    Defaults to "PROD" when the value is absent or unrecognised.
    """
    if raw is None or str(raw).strip() == "":
        return "PROD"
    upper = str(raw).strip().upper()
    if upper in ACTIVITY_TYPES:
        return upper
    return "PROD"


def _detect_data_start(df_raw: pd.DataFrame) -> int:
    """Return the row index where actual tabular data begins.

    Unanet pivot exports often have 1–3 merged header rows that contain report
    titles and date ranges rather than column names. We detect the first row
    where at least 3 cells are non-empty strings that look like column headers
    (i.e. they are not purely numeric).
    """
    for i, row in df_raw.iterrows():
        non_empty = [c for c in row if pd.notna(c) and str(c).strip()]
        non_numeric = [v for v in non_empty if not str(v).replace(".", "").replace("-", "").isnumeric()]
        if len(non_numeric) >= 3:
            return int(i)  # type: ignore[arg-type]
    return 0


# ---------------------------------------------------------------------------
# Public parse functions
# ---------------------------------------------------------------------------

def parse_csv(file_path: Path) -> pd.DataFrame:
    """Read a CSV file and return a normalized DataFrame.

    Column names are lowercased and stripped. Common column aliases are not
    resolved here — that happens in ``_resolve_columns`` later so that the
    raw column names are visible for debugging.

    Raises
    ------
    ValueError
        If the file cannot be parsed or yields an empty DataFrame.
    """
    encodings = ["utf-8-sig", "utf-8", "latin-1"]
    last_err: Exception | None = None
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc, dtype=str)
            break
        except UnicodeDecodeError as exc:
            last_err = exc
    else:
        raise ValueError(f"Cannot decode {file_path.name}: {last_err}") from last_err

    # Normalize column names in-place
    df.columns = [_normalize_header(c) for c in df.columns]
    df.dropna(how="all", inplace=True)

    if df.empty:
        raise ValueError(f"{file_path.name} contains no data rows after parsing.")

    return df


def parse_xlsx(file_path: Path) -> pd.DataFrame:
    """Read an Excel file, skipping merged/title header rows.

    Uses openpyxl engine. The function:
    1. Loads the first sheet without a header row to detect where data starts.
    2. Re-reads from the detected header row onward.
    3. Normalizes column names.

    Raises
    ------
    ValueError
        If the file cannot be parsed or yields an empty DataFrame.
    """
    # Load raw without headers to detect the true header row
    df_raw = pd.read_excel(file_path, header=None, engine="openpyxl", dtype=str)
    header_row = _detect_data_start(df_raw)

    # Re-read with the correct header row
    df = pd.read_excel(
        file_path,
        header=header_row,
        engine="openpyxl",
        dtype=str,
    )
    df.columns = [_normalize_header(c) for c in df.columns]

    # Drop fully-empty rows and unnamed placeholder columns (from merged cells)
    unnamed_cols = [c for c in df.columns if c.startswith("unnamed")]
    df.drop(columns=unnamed_cols, inplace=True, errors="ignore")
    df.dropna(how="all", inplace=True)

    if df.empty:
        raise ValueError(f"{file_path.name} contains no data rows after parsing.")

    return df


# ---------------------------------------------------------------------------
# Main import function
# ---------------------------------------------------------------------------

async def import_file(
    file_path: Path,
    source: str,
    user_id: str | None,
    session: AsyncSession,
) -> dict[str, Any]:
    """Parse a CSV or Excel labor hours export and upsert into the database.

    Parameters
    ----------
    file_path:
        Absolute path to the uploaded file.
    source:
        Data-source tag stored on ActualHours; typically "csv_upload".
    user_id:
        ID of the authenticated user performing the import (may be None).
    session:
        Active async SQLAlchemy session.  The caller is responsible for
        providing and closing this session; ``import_file`` commits at the end.

    Returns
    -------
    dict with keys:
        import_id, status, rows_imported, rows_skipped, periods, programs, warnings
    """
    warnings: list[str] = []
    rows_imported = 0
    rows_skipped = 0
    periods_seen: set[str] = set()
    programs_seen: set[str] = set()

    # ------------------------------------------------------------------
    # 1. Parse raw file
    # ------------------------------------------------------------------
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".csv":
            df = parse_csv(file_path)
        elif suffix in (".xlsx", ".xls"):
            df = parse_xlsx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix!r}. Expected .csv or .xlsx.")
    except Exception as exc:
        # Create a failed DataImport record before re-raising
        import_record = DataImport(
            id=str(uuid.uuid4()),
            source=source,
            filename=file_path.name,
            rows_imported=0,
            rows_skipped=0,
            imported_by=user_id,
            status="failed",
            error_log=str(exc),
        )
        session.add(import_record)
        await session.commit()
        return {
            "import_id": import_record.id,
            "status": "failed",
            "rows_imported": 0,
            "rows_skipped": 0,
            "periods": [],
            "programs": [],
            "warnings": [str(exc)],
        }

    # ------------------------------------------------------------------
    # 2. Resolve column mapping
    # ------------------------------------------------------------------
    col_map = _resolve_columns(df)

    required_cols = ["program", "period", "cost_pool", "hours"]
    missing = [c for c in required_cols if c not in col_map]
    if missing:
        available = list(df.columns)
        msg = (
            f"Required columns not found: {missing}. "
            f"Available columns: {available}. "
            "Check COLUMN_ALIASES for supported header names."
        )
        import_record = DataImport(
            id=str(uuid.uuid4()),
            source=source,
            filename=file_path.name,
            rows_imported=0,
            rows_skipped=0,
            imported_by=user_id,
            status="failed",
            error_log=msg,
        )
        session.add(import_record)
        await session.commit()
        return {
            "import_id": import_record.id,
            "status": "failed",
            "rows_imported": 0,
            "rows_skipped": 0,
            "periods": [],
            "programs": [],
            "warnings": [msg],
        }

    # ------------------------------------------------------------------
    # 3. Pre-load lookup caches (programs and periods) to avoid N+1 queries
    # ------------------------------------------------------------------
    all_programs: dict[str, Program] = {}
    result = await session.execute(select(Program))
    for prog in result.scalars().all():
        all_programs[prog.code.upper()] = prog

    all_periods: dict[str, ForecastPeriod] = {}
    result = await session.execute(select(ForecastPeriod))
    for period in result.scalars().all():
        all_periods[period.label.upper()] = period

    # ------------------------------------------------------------------
    # 4. Pre-load existing ActualHours for upsert keying
    # ------------------------------------------------------------------
    existing_actuals: dict[tuple[str, str, str, str], ActualHours] = {}
    result = await session.execute(select(ActualHours))
    for ah in result.scalars().all():
        key = (ah.program_id, ah.period_id, ah.cost_pool, ah.activity_type)
        existing_actuals[key] = ah

    # ------------------------------------------------------------------
    # 5. Create the DataImport audit record (will update status at end)
    # ------------------------------------------------------------------
    import_id = str(uuid.uuid4())
    import_record = DataImport(
        id=import_id,
        source=source,
        filename=file_path.name,
        rows_imported=0,
        rows_skipped=0,
        imported_by=user_id,
        status="success",
    )
    session.add(import_record)

    # ------------------------------------------------------------------
    # 6. Process rows
    # ------------------------------------------------------------------
    # We group by (program, period, cost_pool, activity_type) and aggregate
    # hours + distinct person count, because the CSV may have one row per
    # person-month rather than one row per group.
    GroupKey = tuple[str, str, str, str]  # (prog_code, period_label, cost_pool, activity_type)

    group_hours: dict[GroupKey, Decimal] = {}
    group_persons: dict[GroupKey, set[str]] = {}
    group_rate: dict[GroupKey, Decimal | None] = {}
    row_warnings: list[str] = []

    for idx, row in df.iterrows():
        row_num = int(idx) + 2  # 1-based, accounting for header

        # --- Program ---
        raw_program = str(row[col_map["program"]]).strip()
        if not raw_program or raw_program.lower() in ("nan", "none", ""):
            row_warnings.append(f"Row {row_num}: empty program code — skipped.")
            rows_skipped += 1
            continue

        prog_code = raw_program.upper()
        if prog_code not in all_programs:
            row_warnings.append(
                f"Row {row_num}: unknown program code {raw_program!r} — skipped. "
                "Create the program first in the Programs admin page."
            )
            rows_skipped += 1
            continue

        # --- Period ---
        raw_period = str(row[col_map["period"]]).strip()
        period_label = _normalize_period_label(raw_period)
        if period_label is None:
            row_warnings.append(
                f"Row {row_num}: cannot parse period {raw_period!r} — skipped."
            )
            rows_skipped += 1
            continue

        if period_label.upper() not in all_periods:
            row_warnings.append(
                f"Row {row_num}: period {period_label!r} not found in forecast_periods — skipped. "
                "Seed ForecastPeriod records for this label first."
            )
            rows_skipped += 1
            continue

        # --- Cost Pool ---
        raw_pool = str(row[col_map["cost_pool"]]).strip()
        cost_pool = _normalize_cost_pool(raw_pool)
        if cost_pool is None:
            row_warnings.append(
                f"Row {row_num}: unrecognised cost_pool {raw_pool!r} — skipped. "
                f"Valid values: {COST_POOLS}."
            )
            rows_skipped += 1
            continue

        # --- Activity Type ---
        raw_activity = (
            str(row[col_map["activity_type"]]).strip()
            if "activity_type" in col_map
            else None
        )
        activity_type = _normalize_activity_type(raw_activity)

        # --- Hours ---
        raw_hours = str(row[col_map["hours"]]).strip()
        try:
            hours_val = Decimal(raw_hours.replace(",", ""))
        except Exception:
            row_warnings.append(
                f"Row {row_num}: cannot parse hours value {raw_hours!r} — skipped."
            )
            rows_skipped += 1
            continue

        if hours_val < 0:
            row_warnings.append(
                f"Row {row_num}: negative hours ({hours_val}) — skipped."
            )
            rows_skipped += 1
            continue

        # --- Rate (optional) ---
        rate_val: Decimal | None = None
        if "rate" in col_map:
            raw_rate = str(row[col_map["rate"]]).strip()
            if raw_rate and raw_rate.lower() not in ("nan", "none", ""):
                try:
                    rate_val = Decimal(raw_rate.replace(",", "").replace("$", ""))
                except Exception:
                    row_warnings.append(
                        f"Row {row_num}: cannot parse rate {raw_rate!r} — rate ignored."
                    )

        # --- Person (optional, used for headcount) ---
        person_name: str | None = None
        if "person" in col_map:
            pname = str(row[col_map["person"]]).strip()
            if pname and pname.lower() not in ("nan", "none", ""):
                person_name = pname

        # --- Accumulate into group ---
        group_key: GroupKey = (prog_code, period_label.upper(), cost_pool, activity_type)
        group_hours[group_key] = group_hours.get(group_key, Decimal("0")) + hours_val
        if group_key not in group_persons:
            group_persons[group_key] = set()
        if person_name:
            group_persons[group_key].add(person_name)

        # Store first non-None rate encountered for the group
        if rate_val is not None and group_rate.get(group_key) is None:
            group_rate[group_key] = rate_val

    # Emit row-level warnings (cap at 50 to avoid flooding the response)
    warnings.extend(row_warnings[:50])
    if len(row_warnings) > 50:
        warnings.append(
            f"... and {len(row_warnings) - 50} more row warnings (truncated)."
        )

    # ------------------------------------------------------------------
    # 7. Upsert ActualHours for each group
    # ------------------------------------------------------------------
    for group_key, total_hours in group_hours.items():
        prog_code, period_label, cost_pool, activity_type = group_key

        program = all_programs[prog_code]
        period = all_periods[period_label]
        headcount = max(len(group_persons.get(group_key, set())), 1)
        rate = group_rate.get(group_key)
        total_cost: Decimal | None = (
            (total_hours * rate).quantize(Decimal("0.01")) if rate is not None else None
        )

        db_key = (program.id, period.id, cost_pool, activity_type)

        if db_key in existing_actuals:
            # Update existing record
            ah = existing_actuals[db_key]
            ah.total_hours = total_hours
            ah.headcount = headcount
            if total_cost is not None:
                ah.total_cost = total_cost
            ah.source = source
            ah.import_id = import_id
        else:
            # Insert new record
            ah = ActualHours(
                id=str(uuid.uuid4()),
                program_id=program.id,
                period_id=period.id,
                cost_pool=cost_pool,
                activity_type=activity_type,
                total_hours=total_hours,
                headcount=headcount,
                total_cost=total_cost,
                source=source,
                import_id=import_id,
            )
            session.add(ah)
            existing_actuals[db_key] = ah  # prevent dupe within same import

        programs_seen.add(prog_code)
        periods_seen.add(period_label)
        rows_imported += 1

    # ------------------------------------------------------------------
    # 8. Finalise the DataImport audit record
    # ------------------------------------------------------------------
    final_status = "success"
    if rows_imported == 0:
        final_status = "failed"
    elif rows_skipped > 0:
        final_status = "partial"

    import_record.rows_imported = rows_imported
    import_record.rows_skipped = rows_skipped
    import_record.periods_covered = json.dumps(sorted(periods_seen))
    import_record.programs_covered = json.dumps(sorted(programs_seen))
    import_record.status = final_status
    if warnings:
        import_record.error_log = "\n".join(warnings)

    await session.commit()

    logger.info(
        "import_file completed: file=%s status=%s imported=%d skipped=%d",
        file_path.name,
        final_status,
        rows_imported,
        rows_skipped,
    )

    return {
        "import_id": import_id,
        "status": final_status,
        "rows_imported": rows_imported,
        "rows_skipped": rows_skipped,
        "periods": sorted(periods_seen),
        "programs": sorted(programs_seen),
        "warnings": warnings,
    }
