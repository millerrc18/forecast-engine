"""Data import page and API routes."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select

from forecast_engine.auth.middleware import get_current_user, require_auth
from forecast_engine.config import settings
from forecast_engine.models.actuals import DataImport
from forecast_engine.models.base import async_session
from forecast_engine.services.importer import import_file
from forecast_engine.templating import templates

# ---------------------------------------------------------------------------
# Page router — HTML views
# ---------------------------------------------------------------------------

page_router = APIRouter(tags=["import-pages"])


@page_router.get("/import", response_class=HTMLResponse)
async def import_page(request: Request):
    """Data import management page (PM and admin)."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/auth/login", status_code=302)
    if user.get("role") not in ("pm", "admin"):
        return RedirectResponse("/dashboard", status_code=302)

    async with async_session() as session:
        result = await session.execute(
            select(DataImport)
            .order_by(DataImport.imported_at.desc())
            .limit(50)
        )
        imports = result.scalars().all()

    return templates.TemplateResponse(request, "import.html", {
        "user": user,
        "imports": imports,
    })


# ---------------------------------------------------------------------------
# API router — JSON endpoints
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/import", tags=["import"])


@router.post("/csv")
async def upload_csv(request: Request, file: UploadFile):
    """Accept a CSV or Excel file upload and run the importer."""
    user = require_auth(request)

    # Ensure upload directory exists
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename and write to disk
    safe_name = Path(file.filename).name if file.filename else "upload"
    dest = settings.upload_dir / safe_name

    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    async with async_session() as session:
        result = await import_file(dest, "csv_upload", user["id"], session)

    return result


@router.get("/history")
async def import_history(request: Request):
    """List recent DataImport audit records."""
    require_auth(request)

    async with async_session() as session:
        db_result = await session.execute(
            select(DataImport)
            .order_by(DataImport.imported_at.desc())
            .limit(100)
        )
        records = db_result.scalars().all()

    return [
        {
            "id": r.id,
            "source": r.source,
            "filename": r.filename,
            "rows_imported": r.rows_imported,
            "rows_skipped": r.rows_skipped,
            "status": r.status,
            "imported_at": r.imported_at.isoformat() if r.imported_at else None,
        }
        for r in records
    ]


@router.post("/unanet")
async def trigger_unanet_pull(request: Request):
    """Trigger a data pull from Unanet MCP."""
    from datetime import date

    from forecast_engine.services.mcp_unanet import pull_unanet_hours

    user = require_auth(request)
    # TODO: Parse date range from request body
    async with async_session() as session:
        result = await pull_unanet_hours(
            start_date=date.today().replace(day=1),
            end_date=date.today(),
            session=session,
            user_id=user["id"],
        )
    return result


@router.post("/oracle")
async def trigger_oracle_pull(request: Request):
    """Trigger a data pull from Oracle MCP."""
    from forecast_engine.services.mcp_oracle import pull_oracle_data

    user = require_auth(request)
    async with async_session() as session:
        result = await pull_oracle_data(
            session=session,
            user_id=user["id"],
        )
    return result
