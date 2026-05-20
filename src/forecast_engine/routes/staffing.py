"""Staffing allocation page and API routes — functional manager responses to demand signals."""

from __future__ import annotations

from decimal import Decimal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from forecast_engine.auth.middleware import get_current_user, require_auth
from forecast_engine.models.base import async_session
from forecast_engine.models.demand import DemandSignal, StaffingAllocation
from forecast_engine.models.program import Program
from forecast_engine.templating import templates

# ---------------------------------------------------------------------------
# Page router — HTML views
# ---------------------------------------------------------------------------

page_router = APIRouter(tags=["staffing-pages"])


@page_router.get("/staffing", response_class=HTMLResponse)
async def staffing_page(request: Request):
    """Staffing allocation page for functional managers."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/auth/login", status_code=302)

    async with async_session() as session:
        # Fetch demand signals with status "submitted" — these need staffing responses
        signals_q = (
            select(DemandSignal)
            .where(DemandSignal.status == "submitted")
            .options(
                selectinload(DemandSignal.program),
                selectinload(DemandSignal.submitter),
                selectinload(DemandSignal.allocations),
            )
            .order_by(DemandSignal.period_start.desc())
        )
        signals_result = await session.execute(signals_q)
        pending_signals = signals_result.scalars().all()

        # Fetch existing allocations submitted by this user
        my_allocs_q = (
            select(StaffingAllocation)
            .where(StaffingAllocation.submitted_by == user["id"])
            .options(
                selectinload(StaffingAllocation.demand_signal),
                selectinload(StaffingAllocation.program),
            )
            .order_by(StaffingAllocation.created_at.desc())
        )
        my_allocs_result = await session.execute(my_allocs_q)
        my_allocations = my_allocs_result.scalars().all()

    return templates.TemplateResponse(
        request,
        "staffing.html",
        {
            "user": user,
            "pending_signals": pending_signals,
            "my_allocations": my_allocations,
        },
    )


# ---------------------------------------------------------------------------
# API router — JSON endpoints
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api", tags=["staffing"])


# -- Pydantic schemas --------------------------------------------------------

class AllocationCreate(BaseModel):
    cost_pool: str
    fte_count: float
    avg_allocation_pct: float
    blended_rate: float | None = None
    notes: str | None = None


class AllocationUpdate(BaseModel):
    cost_pool: str | None = None
    fte_count: float | None = None
    avg_allocation_pct: float | None = None
    blended_rate: float | None = None
    notes: str | None = None


# -- Helpers -----------------------------------------------------------------

def _alloc_to_dict(alloc: StaffingAllocation) -> dict:
    return {
        "id": alloc.id,
        "demand_signal_id": alloc.demand_signal_id,
        "program_id": alloc.program_id,
        "submitted_by": alloc.submitted_by,
        "cost_pool": alloc.cost_pool,
        "fte_count": float(alloc.fte_count),
        "avg_allocation_pct": float(alloc.avg_allocation_pct),
        "blended_rate": float(alloc.blended_rate) if alloc.blended_rate is not None else None,
        "planned_hrs_per_month": float(alloc.planned_hrs_per_month) if alloc.planned_hrs_per_month is not None else None,
        "notes": alloc.notes,
        "status": alloc.status,
        "created_at": alloc.created_at.isoformat() if alloc.created_at else None,
    }


def _compute_planned_hrs(fte_count: float, avg_allocation_pct: float) -> Decimal:
    """Compute planned hours per month: FTEs * (alloc% / 100) * 168."""
    return Decimal(str(round(fte_count * (avg_allocation_pct / 100) * 168, 1)))


# -- Routes ------------------------------------------------------------------

@router.get("/demand-signals/{signal_id}/allocations")
async def list_allocations(signal_id: str, request: Request):
    """List all staffing allocations for a demand signal."""
    user = require_auth(request)

    async with async_session() as session:
        signal = await session.get(DemandSignal, signal_id)
        if not signal:
            raise HTTPException(status_code=404, detail="Demand signal not found")

        result = await session.execute(
            select(StaffingAllocation)
            .where(StaffingAllocation.demand_signal_id == signal_id)
            .order_by(StaffingAllocation.created_at)
        )
        allocations = result.scalars().all()

    return [_alloc_to_dict(a) for a in allocations]


@router.post("/demand-signals/{signal_id}/allocations", status_code=201)
async def create_allocation(
    signal_id: str,
    body: AllocationCreate,
    request: Request,
):
    """Create a staffing allocation in response to a demand signal."""
    user = require_auth(request)

    if body.cost_pool not in ("BAMTL", "ENGTL"):
        raise HTTPException(status_code=422, detail="cost_pool must be BAMTL or ENGTL")
    if body.fte_count <= 0:
        raise HTTPException(status_code=422, detail="fte_count must be positive")
    if not (0 < body.avg_allocation_pct <= 100):
        raise HTTPException(status_code=422, detail="avg_allocation_pct must be between 0 and 100")

    async with async_session() as session:
        signal = await session.get(DemandSignal, signal_id)
        if not signal:
            raise HTTPException(status_code=404, detail="Demand signal not found")

        if signal.status not in ("submitted", "acknowledged"):
            raise HTTPException(
                status_code=409,
                detail=f"Cannot allocate against a signal with status '{signal.status}'",
            )

        planned_hrs = _compute_planned_hrs(body.fte_count, body.avg_allocation_pct)

        alloc = StaffingAllocation(
            demand_signal_id=signal_id,
            program_id=signal.program_id,
            submitted_by=user["id"],
            cost_pool=body.cost_pool,
            fte_count=Decimal(str(body.fte_count)),
            avg_allocation_pct=Decimal(str(body.avg_allocation_pct)),
            blended_rate=Decimal(str(body.blended_rate)) if body.blended_rate is not None else None,
            planned_hrs_per_month=planned_hrs,
            notes=body.notes,
            status="draft",
        )
        session.add(alloc)
        await session.commit()
        await session.refresh(alloc)

    return _alloc_to_dict(alloc)


@router.put("/allocations/{alloc_id}")
async def update_allocation(
    alloc_id: str,
    body: AllocationUpdate,
    request: Request,
):
    """Update a draft staffing allocation."""
    user = require_auth(request)

    async with async_session() as session:
        alloc = await session.get(StaffingAllocation, alloc_id)
        if not alloc:
            raise HTTPException(status_code=404, detail="Allocation not found")

        if alloc.submitted_by != user["id"] and user["role"] not in ("admin",):
            raise HTTPException(status_code=403, detail="Not authorized to edit this allocation")

        if alloc.status != "draft":
            raise HTTPException(
                status_code=409,
                detail=f"Cannot edit an allocation with status '{alloc.status}'",
            )

        if body.cost_pool is not None:
            if body.cost_pool not in ("BAMTL", "ENGTL"):
                raise HTTPException(status_code=422, detail="cost_pool must be BAMTL or ENGTL")
            alloc.cost_pool = body.cost_pool
        if body.fte_count is not None:
            alloc.fte_count = Decimal(str(body.fte_count))
        if body.avg_allocation_pct is not None:
            alloc.avg_allocation_pct = Decimal(str(body.avg_allocation_pct))
        if body.blended_rate is not None:
            alloc.blended_rate = Decimal(str(body.blended_rate))
        if body.notes is not None:
            alloc.notes = body.notes

        # Recompute planned hours whenever FTE or allocation % changes
        alloc.planned_hrs_per_month = _compute_planned_hrs(
            float(alloc.fte_count), float(alloc.avg_allocation_pct)
        )

        await session.commit()
        await session.refresh(alloc)

    return _alloc_to_dict(alloc)


@router.post("/allocations/{alloc_id}/submit")
async def submit_allocation(alloc_id: str, request: Request):
    """Advance a draft allocation to submitted status."""
    user = require_auth(request)

    async with async_session() as session:
        alloc = await session.get(StaffingAllocation, alloc_id)
        if not alloc:
            raise HTTPException(status_code=404, detail="Allocation not found")

        if alloc.submitted_by != user["id"] and user["role"] not in ("admin",):
            raise HTTPException(status_code=403, detail="Not authorized to submit this allocation")

        if alloc.status != "draft":
            raise HTTPException(
                status_code=409,
                detail=f"Allocation is already '{alloc.status}', cannot submit",
            )

        alloc.status = "submitted"
        await session.commit()
        await session.refresh(alloc)

    return _alloc_to_dict(alloc)


@router.get("/func-mgr/pending")
async def pending_demand_signals(request: Request):
    """List demand signals awaiting a staffing response from the current user."""
    user = require_auth(request)

    async with async_session() as session:
        # All submitted signals
        signals_q = (
            select(DemandSignal)
            .where(DemandSignal.status == "submitted")
            .options(
                selectinload(DemandSignal.program),
                selectinload(DemandSignal.submitter),
                selectinload(DemandSignal.allocations),
            )
            .order_by(DemandSignal.period_start.desc())
        )
        result = await session.execute(signals_q)
        signals = result.scalars().all()

    return [
        {
            "id": s.id,
            "program_id": s.program_id,
            "program_code": s.program.code if s.program else None,
            "program_name": s.program.name if s.program else None,
            "period_start": str(s.period_start),
            "period_end": str(s.period_end),
            "units_in_flow": s.units_in_flow,
            "status": s.status,
            "allocation_count": len(s.allocations),
            "my_allocation_count": sum(
                1 for a in s.allocations if a.submitted_by == user["id"]
            ),
        }
        for s in signals
    ]
