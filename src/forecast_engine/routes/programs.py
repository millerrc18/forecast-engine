"""Program API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from forecast_engine.auth.middleware import require_auth
from forecast_engine.models.base import async_session
from forecast_engine.models.program import Program

router = APIRouter(prefix="/api/programs", tags=["programs"])


@router.get("")
async def list_programs(user: dict = Depends(require_auth)):
    async with async_session() as session:
        query = select(Program).order_by(Program.code)

        if user["role"] == "pm":
            query = query.where(Program.pm_user_id == user["id"])

        result = await session.execute(query)
        programs = result.scalars().all()

    return [
        {
            "id": p.id,
            "code": p.code,
            "name": p.name,
            "status": p.status,
            "site": p.site,
        }
        for p in programs
    ]


@router.get("/{program_id}")
async def get_program(program_id: str, user: dict = Depends(require_auth)):
    async with async_session() as session:
        program = await session.get(
            Program, program_id, options=[selectinload(Program.pm_user)]
        )

    if not program:
        raise HTTPException(status_code=404, detail="Program not found")

    return {
        "id": program.id,
        "code": program.code,
        "name": program.name,
        "status": program.status,
        "site": program.site,
        "contract_type": program.contract_type,
        "pm": program.pm_user.display_name if program.pm_user else None,
        "pop_start": str(program.pop_start) if program.pop_start else None,
        "pop_end": str(program.pop_end) if program.pop_end else None,
    }
