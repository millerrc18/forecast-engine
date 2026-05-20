"""Forecast generation API routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select

from forecast_engine.auth.middleware import require_auth
from forecast_engine.models.base import async_session
from forecast_engine.models.program import ForecastPeriod, Program
from forecast_engine.services.forecast import ForecastService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/forecasts", tags=["forecasts"])


class GenerateRequest(BaseModel):
    period_label: str | None = None  # e.g. "JUN-26"; defaults to latest period


@router.post("/generate")
async def generate_forecasts(request: Request, body: GenerateRequest | None = None):
    """Generate Tier-1 (headcount) + Tier-2 (GBM) forecasts for all programs."""
    user = require_auth(request)

    if user["role"] not in ("pm", "admin"):
        raise HTTPException(status_code=403, detail="Only PMs and admins can generate forecasts")

    async with async_session() as session:
        # Resolve period
        if body and body.period_label:
            result = await session.execute(
                select(ForecastPeriod).where(ForecastPeriod.label == body.period_label)
            )
            period = result.scalars().first()
            if not period:
                raise HTTPException(status_code=404, detail=f"Period '{body.period_label}' not found")
        else:
            # Use the latest period
            result = await session.execute(
                select(ForecastPeriod).order_by(ForecastPeriod.start_date.desc()).limit(1)
            )
            period = result.scalars().first()
            if not period:
                raise HTTPException(status_code=404, detail="No forecast periods found")

        # Scope programs: PMs see only theirs, admins see all
        if user["role"] == "pm":
            prog_result = await session.execute(
                select(Program).where(
                    Program.status == "active",
                    Program.pm_user_id == user["id"],
                )
            )
        else:
            prog_result = await session.execute(
                select(Program).where(Program.status == "active")
            )
        programs = prog_result.scalars().all()

        if not programs:
            return {"period": period.label, "results": [], "message": "No active programs found"}

        svc = ForecastService()
        results = []
        errors = []

        for prog in programs:
            try:
                res = await svc.generate_forecast(session, prog.id, period.id)
                ravg = res.get("rolling_avg")
                hc = res.get("headcount")
                gbm = res.get("gbm")
                results.append({
                    "program": prog.code,
                    "rolling_avg_hrs": float(ravg.predicted_support_hrs) if ravg else None,
                    "headcount_hrs": float(hc.predicted_support_hrs) if hc else None,
                    "gbm_hrs": float(gbm.predicted_support_hrs) if gbm else None,
                    "sanity": res.get("sanity_check"),
                })
            except Exception as e:
                logger.exception("Forecast failed for %s", prog.code)
                errors.append({"program": prog.code, "error": str(e)})

        await session.commit()

    return {
        "period": period.label,
        "generated": len(results),
        "errors": len(errors),
        "results": results,
        "error_details": errors if errors else None,
    }
