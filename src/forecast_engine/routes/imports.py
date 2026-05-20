"""Data import page and API routes."""

from __future__ import annotations

import asyncio
import uuid as _uuid
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

_ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
_ALLOWED_MIMES = {
    "text/csv",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/octet-stream",  # browsers sometimes send this
}

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

    # Validate file extension
    original_name = Path(file.filename).name if file.filename else "upload"
    ext = Path(original_name).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(_ALLOWED_EXTENSIONS)}",
        )

    # Validate MIME type
    if file.content_type and file.content_type not in _ALLOWED_MIMES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{file.content_type}'.",
        )

    # Ensure upload directory exists
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    # UUID-prefixed filename to prevent overwrites and path traversal
    safe_name = f"{_uuid.uuid4().hex[:12]}_{original_name}"
    dest = settings.upload_dir / safe_name

    # Stream to disk with size limit (runs blocking I/O in thread)
    max_bytes = settings.max_upload_mb * 1024 * 1024

    def _write_with_limit():
        total = 0
        with dest.open("wb") as out:
            while True:
                chunk = file.file.read(64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {settings.max_upload_mb} MB limit.",
                    )
                out.write(chunk)

    await asyncio.to_thread(_write_with_limit)

    try:
        async with async_session() as session:
            result = await import_file(dest, "csv_upload", user["id"], session)
    finally:
        # Clean up uploaded file after import
        dest.unlink(missing_ok=True)

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
