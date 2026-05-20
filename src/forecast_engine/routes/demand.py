"""Demand signal page and API routes."""

from __future__ import annotations

import json
from datetime import date

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from forecast_engine.auth.middleware import get_current_user, require_auth
from forecast_engine.models.base import async_session
from forecast_engine.models.demand import DemandSignal
from forecast_engine.models.program import Program
from forecast_engine.templating import templates

# ---------------------------------------------------------------------------
# Page router — HTML views
# ---------------------------------------------------------------------------

page_router = APIRouter(tags=["demand-pages"])


@page_router.get("/demand-signals", response_class=HTMLResponse)
async def demand_signals_page(request: Request):
    """Demand signal management page."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/auth/login", status_code=302)

    async with async_session() as session:
        # Fetch demand signals, optionally scoped to PM's programs
        query = (
            select(DemandSignal)
            .options(
                selectinload(DemandSignal.program),
                selectinload(DemandSignal.submitter),
            )
            .order_by(DemandSignal.period_start.desc())
        )

        if user["role"] == "pm":
            # Scope to signals for programs owned by this PM
            pm_programs_q = select(Program.id).where(
                Program.pm_user_id == user["id"]
            )
            query = query.where(DemandSignal.program_id.in_(pm_programs_q))

        result = await session.execute(query)
        signals = result.scalars().all()

        # Fetch programs for the "New Signal" form dropdown
        prog_query = select(Program).order_by(Program.code)
        if user["role"] == "pm":
            prog_query = prog_query.where(Program.pm_user_id == user["id"])
        prog_result = await session.execute(prog_query)
        programs = prog_result.scalars().all()

    return templates.TemplateResponse(
        request,
        "demand_signal.html",
        {
            "user": user,
            "signals": signals,
            "programs": programs,
        },
    )


# ---------------------------------------------------------------------------
# API router — JSON endpoints
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api", tags=["demand"])


# -- Pydantic schemas --------------------------------------------------------

class DemandSignalCreate(BaseModel):
    period_start: date
    period_end: date
    units_in_flow: int | None = None
    milestones: list[dict] | None = None  # [{"name", "target_date", "type"}]
    scope_changes: str | None = None
    notes: str | None = None


class DemandSignalUpdate(BaseModel):
    period_start: date | None = None
    period_end: date | None = None
    units_in_flow: int | None = None
    milestones: list[dict] | None = None
    scope_changes: str | None = None
    notes: str | None = None


# -- Helpers -----------------------------------------------------------------

def _signal_to_dict(signal: DemandSignal) -> dict:
    return {
        "id": signal.id,
        "program_id": signal.program_id,
        "submitted_by": signal.submitted_by,
        "period_start": str(signal.period_start),
        "period_end": str(signal.period_end),
        "units_in_flow": signal.units_in_flow,
        "milestones": json.loads(signal.milestones) if signal.milestones else [],
        "scope_changes": signal.scope_changes,
        "notes": signal.notes,
        "status": signal.status,
        "created_at": signal.created_at.isoformat() if signal.created_at else None,
        "updated_at": signal.updated_at.isoformat() if signal.updated_at else None,
    }


# -- Routes ------------------------------------------------------------------

@router.get("/programs/{program_id}/demand-signals")
async def list_demand_signals(program_id: str, request: Request):
    """List demand signals for a program."""
    user = require_auth(request)

    async with async_session() as session:
        program = await session.get(Program, program_id)
        if not program:
            raise HTTPException(status_code=404, detail="Program not found")

        # PMs can only view signals for their own programs
        if user["role"] == "pm" and program.pm_user_id != user["id"]:
            raise HTTPException(status_code=403, detail="Not authorized for this program")

        result = await session.execute(
            select(DemandSignal)
            .where(DemandSignal.program_id == program_id)
            .order_by(DemandSignal.period_start.desc())
        )
        signals = result.scalars().all()

    return [_signal_to_dict(s) for s in signals]


@router.post("/programs/{program_id}/demand-signals", status_code=201)
async def create_demand_signal(
    program_id: str,
    body: DemandSignalCreate,
    request: Request,
):
    """Create a new demand signal (draft) for a program."""
    user = require_auth(request)

    async with async_session() as session:
        program = await session.get(Program, program_id)
        if not program:
            raise HTTPException(status_code=404, detail="Program not found")

        if user["role"] == "pm" and program.pm_user_id != user["id"]:
            raise HTTPException(status_code=403, detail="Not authorized for this program")

        signal = DemandSignal(
            program_id=program_id,
            submitted_by=user["id"],
            period_start=body.period_start,
            period_end=body.period_end,
            units_in_flow=body.units_in_flow,
            milestones=json.dumps(body.milestones) if body.milestones else None,
            scope_changes=body.scope_changes,
            notes=body.notes,
            status="draft",
        )
        session.add(signal)
        await session.commit()
        await session.refresh(signal)

    return _signal_to_dict(signal)


@router.put("/demand-signals/{signal_id}")
async def update_demand_signal(
    signal_id: str,
    body: DemandSignalUpdate,
    request: Request,
):
    """Update a demand signal (only allowed when status == draft and user owns it)."""
    user = require_auth(request)

    async with async_session() as session:
        signal = await session.get(DemandSignal, signal_id)
        if not signal:
            raise HTTPException(status_code=404, detail="Demand signal not found")

        if signal.submitted_by != user["id"] and user["role"] not in ("admin", "leadership"):
            raise HTTPException(status_code=403, detail="Not authorized to edit this signal")

        if signal.status != "draft":
            raise HTTPException(
                status_code=409,
                detail=f"Cannot edit a signal with status '{signal.status}'",
            )

        if body.period_start is not None:
            signal.period_start = body.period_start
        if body.period_end is not None:
            signal.period_end = body.period_end
        if body.units_in_flow is not None:
            signal.units_in_flow = body.units_in_flow
        if body.milestones is not None:
            signal.milestones = json.dumps(body.milestones)
        if body.scope_changes is not None:
            signal.scope_changes = body.scope_changes
        if body.notes is not None:
            signal.notes = body.notes

        await session.commit()
        await session.refresh(signal)

    return _signal_to_dict(signal)


@router.post("/demand-signals/{signal_id}/submit")
async def submit_demand_signal(signal_id: str, request: Request):
    """Advance a draft demand signal to submitted status."""
    user = require_auth(request)

    async with async_session() as session:
        signal = await session.get(DemandSignal, signal_id)
        if not signal:
            raise HTTPException(status_code=404, detail="Demand signal not found")

        if signal.submitted_by != user["id"] and user["role"] not in ("admin",):
            raise HTTPException(status_code=403, detail="Not authorized to submit this signal")

        if signal.status != "draft":
            raise HTTPException(
                status_code=409,
                detail=f"Signal is already '{signal.status}', cannot submit",
            )

        signal.status = "submitted"
        await session.commit()
        await session.refresh(signal)

    return _signal_to_dict(signal)


@router.delete("/demand-signals/{signal_id}", status_code=200)
async def delete_demand_signal(signal_id: str, request: Request):
    """Delete a demand signal. Admins can delete any; PMs can delete their own drafts."""
    user = require_auth(request)

    async with async_session() as session:
        signal = await session.get(DemandSignal, signal_id)
        if not signal:
            raise HTTPException(status_code=404, detail="Demand signal not found")

        is_owner = signal.submitted_by == user["id"]
        is_admin = user["role"] in ("admin",)

        if not is_owner and not is_admin:
            raise HTTPException(status_code=403, detail="Not authorized to delete this signal")

        if not is_admin and signal.status != "draft":
            raise HTTPException(
                status_code=409,
                detail="Only admins can delete submitted signals",
            )

        await session.delete(signal)
        await session.commit()

    return {"deleted": signal_id}
