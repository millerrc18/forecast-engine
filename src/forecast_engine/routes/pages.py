"""HTML page routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from forecast_engine.templating import templates
from forecast_engine.auth.middleware import get_current_user
from forecast_engine.models.actuals import ActualHours
from forecast_engine.models.base import async_session
from forecast_engine.models.forecast import Forecast
from forecast_engine.models.program import ForecastPeriod, Program, User

router = APIRouter(tags=["pages"])


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/auth/login", status_code=302)
    return RedirectResponse("/dashboard", status_code=302)


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/auth/login", status_code=302)

    async with async_session() as session:
        query = select(Program).options(selectinload(Program.pm_user)).order_by(Program.code)

        # PMs see only their programs; leadership/admin see all
        if user["role"] == "pm":
            query = query.where(Program.pm_user_id == user["id"])

        result = await session.execute(query)
        programs = result.scalars().all()

        # Check if any actuals exist (drives "Getting Started" banner)
        actuals_count = await session.execute(
            select(ActualHours.id).limit(1)
        )
        has_actuals = actuals_count.scalar_one_or_none() is not None

        # Derive current period from DB
        latest_period_result = await session.execute(
            select(ForecastPeriod).order_by(ForecastPeriod.start_date.desc()).limit(1)
        )
        latest_period = latest_period_result.scalars().first()
        current_period = latest_period.label if latest_period else "N/A"

    return templates.TemplateResponse(request, "dashboard.html", {
        "user": user,
        "programs": programs,
        "current_period": current_period,
        "has_actuals": has_actuals,
    })


@router.get("/programs/{program_id}", response_class=HTMLResponse)
async def program_detail(request: Request, program_id: str):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/auth/login", status_code=302)

    async with async_session() as session:
        program = await session.get(
            Program, program_id, options=[selectinload(Program.pm_user)]
        )

    if not program:
        return HTMLResponse("Program not found", status_code=404)

    return templates.TemplateResponse(request, "program_detail.html", {
        "user": user,
        "program": program,
    })


@router.get("/partials/forecast-card/{program_id}", response_class=HTMLResponse)
async def forecast_card_partial(request: Request, program_id: str):
    """HTMX partial — single program forecast card."""
    user = get_current_user(request)
    if not user:
        return HTMLResponse("", status_code=401)

    async with async_session() as session:
        program = await session.get(
            Program, program_id, options=[selectinload(Program.pm_user)]
        )

        # Get latest period
        period_result = await session.execute(
            select(ForecastPeriod).order_by(ForecastPeriod.start_date.desc()).limit(1)
        )
        latest_period = period_result.scalars().first()

        forecast_data = None
        if latest_period:
            # Load all forecast methods for this program/period
            fc_result = await session.execute(
                select(Forecast).where(
                    Forecast.program_id == program_id,
                    Forecast.period_id == latest_period.id,
                )
            )
            forecasts = {f.method: f for f in fc_result.scalars().all()}

            if forecasts:
                ravg = forecasts.get("rolling_avg")
                hc = forecasts.get("headcount")
                gbm = forecasts.get("gbm")

                # Get latest actual support hours for comparison
                actual_periods = await session.execute(
                    select(ForecastPeriod)
                    .where(ForecastPeriod.start_date < latest_period.start_date)
                    .order_by(ForecastPeriod.start_date.desc())
                    .limit(1)
                )
                last_actual_period = actual_periods.scalars().first()

                actual_hrs = None
                if last_actual_period:
                    ah_result = await session.execute(
                        select(ActualHours).where(
                            ActualHours.program_id == program_id,
                            ActualHours.period_id == last_actual_period.id,
                            ActualHours.cost_pool.in_(["BAMTL", "ENGTL"]),
                        )
                    )
                    actual_hrs = sum(float(a.total_hours) for a in ah_result.scalars().all())

                # Use best available forecast for "primary"
                primary = gbm or hc or ravg
                primary_hrs = float(primary.predicted_support_hrs) if primary else 0

                variance_pct = None
                if actual_hrs and primary_hrs:
                    variance_pct = ((primary_hrs - actual_hrs) / actual_hrs) * 100

                forecast_data = {
                    "rolling_avg_hrs": float(ravg.predicted_support_hrs) if ravg else None,
                    "headcount_hrs": float(hc.predicted_support_hrs) if hc else None,
                    "gbm_hrs": float(gbm.predicted_support_hrs) if gbm else None,
                    "bam_hrs": float(ravg.predicted_bam_hrs) if ravg and ravg.predicted_bam_hrs else None,
                    "eng_hrs": float(ravg.predicted_eng_hrs) if ravg and ravg.predicted_eng_hrs else None,
                    "actual_hrs": actual_hrs,
                    "variance_pct": round(variance_pct, 1) if variance_pct else None,
                    "period_label": latest_period.label,
                }

    return templates.TemplateResponse(request, "partials/forecast_card.html", {
        "user": user,
        "program": program,
        "forecast": forecast_data,
    })


@router.get("/leadership", response_class=HTMLResponse)
async def leadership_dashboard(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/auth/login", status_code=302)
    if user["role"] not in ("leadership", "admin"):
        return RedirectResponse("/dashboard", status_code=302)

    return templates.TemplateResponse(request, "leadership.html", {
        "user": user,
    })
