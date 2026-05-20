"""Feature engineering service for the GBM forecast model.

Computes 19 features from the actuals, periods, programs, and staff tables.
The target variable is ``total_support_hrs`` (BAMTL + ENGTL hours combined).

Public API
----------
FeatureEngineer
    .build_features(session) -> pd.DataFrame        — full training dataset
    .build_features_for_prediction(session, ...) -> dict  — single-row inference
"""

from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from forecast_engine.models.actuals import ActualHours
from forecast_engine.models.program import ForecastPeriod, Program, ProgramStaff

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature name constants — order matches the model's expected input vector.
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    "sup_rolling3",
    "total_support_hc",
    "hrs_per_eng",
    "hrs_per_bam",
    "bam_hc",
    "eng_hc",
    "bam_hrs",
    "eng_hrs",
    "mfg_hrs",
    "sup_lag1",
    "sup_lag2",
    "utilization",
    "quarter",
    "month",
    "program_tenure_months",
    "fte_allocation",
    "pct_bam",
    "pct_eng",
    "is_transition",
]

# Cost pool constants
_BAM = "BAMTL"
_ENG = "ENGTL"
_MFG = "MFGTL"
_SUPPORT_POOLS = {_BAM, _ENG}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(numerator: float, denominator: float) -> float:
    """Divide without raising on zero denominator."""
    return numerator / denominator if denominator != 0 else 0.0


def _months_between(start: date, end: date) -> int:
    """Approximate month count from *start* to *end* (inclusive of partial)."""
    return (end.year - start.year) * 12 + (end.month - start.month)


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """Computes all 19 GBM features from the database."""

    # ------------------------------------------------------------------
    # Public: full training dataset
    # ------------------------------------------------------------------

    async def build_features(self, session: AsyncSession) -> pd.DataFrame:
        """Query all actuals and compute the full feature matrix.

        Returns a DataFrame sorted by (program_id, start_date) with columns:
        ``program_id``, ``period_id``, ``label``, 19 feature columns, and
        ``total_support_hrs`` (the prediction target).
        """
        # 1. Load reference data
        programs = await self._load_programs(session)
        periods = await self._load_periods(session)
        actuals = await self._load_actuals(session)
        staff_fte = await self._load_staff_fte(session, periods)

        if not actuals:
            logger.warning("No actuals found — returning empty DataFrame.")
            return self._empty_dataframe()

        # 2. Build a period ordering lookup (period_id -> sort index)
        period_order = self._build_period_order(periods)

        # 3. Pivot actuals into per-program-period aggregates
        agg = self._aggregate_actuals(actuals)

        # 4. Determine first/last active period per program (for is_transition)
        program_bounds = self._compute_program_bounds(agg, period_order)

        # 5. Assemble rows
        rows: list[dict] = []
        for prog_id, period_map in sorted(agg.items()):
            program = programs.get(prog_id)
            if program is None:
                continue

            # Sort this program's periods chronologically
            sorted_period_ids = sorted(
                period_map.keys(),
                key=lambda pid: period_order.get(pid, 0),
            )

            # Collect support-hours history for lag/rolling computation
            support_history: list[float] = []

            for period_id in sorted_period_ids:
                period = periods.get(period_id)
                if period is None:
                    continue

                data = period_map[period_id]

                # --- raw values ---
                bam_hrs = data.get((_BAM, "hrs"), 0.0)
                eng_hrs = data.get((_ENG, "hrs"), 0.0)
                mfg_hrs = data.get((_MFG, "hrs"), 0.0)
                bam_hc = data.get((_BAM, "hc"), 0)
                eng_hc = data.get((_ENG, "hc"), 0)

                total_support_hrs = bam_hrs + eng_hrs
                total_support_hc = bam_hc + eng_hc

                # --- lag features ---
                n = len(support_history)
                sup_lag1 = support_history[-1] if n >= 1 else 0.0
                sup_lag2 = support_history[-2] if n >= 2 else 0.0

                # --- rolling 3-month average ---
                if n >= 3:
                    sup_rolling3 = sum(support_history[-3:]) / 3.0
                elif n > 0:
                    sup_rolling3 = sum(support_history) / float(n)
                else:
                    sup_rolling3 = total_support_hrs  # first month fallback

                # --- derived ---
                hrs_per_eng = _safe_div(eng_hrs, eng_hc)
                hrs_per_bam = _safe_div(bam_hrs, bam_hc)

                total_pool_hrs = bam_hrs + eng_hrs
                pct_bam = _safe_div(bam_hrs, total_pool_hrs)
                pct_eng = _safe_div(eng_hrs, total_pool_hrs)

                workdays = period.workdays or 22  # fallback
                available_hrs = total_support_hc * workdays * 8
                utilization = _safe_div(total_support_hrs, available_hrs)

                # --- calendar / tenure ---
                quarter = period.quarter
                month = period.start_date.month

                pop_start = program.pop_start
                if pop_start is None:
                    # Use earliest actuals period as fallback
                    first_pid = sorted_period_ids[0]
                    first_period = periods.get(first_pid)
                    pop_start = first_period.start_date if first_period else period.start_date
                program_tenure_months = max(_months_between(pop_start, period.start_date), 0)

                # --- FTE allocation ---
                fte_allocation = staff_fte.get((prog_id, period_id), 0.0)

                # --- is_transition ---
                bounds = program_bounds.get(prog_id)
                if bounds is not None:
                    first_idx, last_idx = bounds
                    current_idx = period_order.get(period_id, 0)
                    is_transition = 1 if (
                        current_idx <= first_idx + 2 or current_idx >= last_idx - 2
                    ) else 0
                else:
                    is_transition = 1

                # --- append row ---
                rows.append({
                    "program_id": prog_id,
                    "period_id": period_id,
                    "label": period.label,
                    # -- features --
                    "sup_rolling3": round(sup_rolling3, 2),
                    "total_support_hc": total_support_hc,
                    "hrs_per_eng": round(hrs_per_eng, 2),
                    "hrs_per_bam": round(hrs_per_bam, 2),
                    "bam_hc": bam_hc,
                    "eng_hc": eng_hc,
                    "bam_hrs": round(bam_hrs, 1),
                    "eng_hrs": round(eng_hrs, 1),
                    "mfg_hrs": round(mfg_hrs, 1),
                    "sup_lag1": round(sup_lag1, 1),
                    "sup_lag2": round(sup_lag2, 1),
                    "utilization": round(utilization, 4),
                    "quarter": quarter,
                    "month": month,
                    "program_tenure_months": program_tenure_months,
                    "fte_allocation": round(fte_allocation, 2),
                    "pct_bam": round(pct_bam, 4),
                    "pct_eng": round(pct_eng, 4),
                    "is_transition": is_transition,
                    # -- target --
                    "total_support_hrs": round(total_support_hrs, 1),
                })

                # Push current support hours into history *after* computing features
                support_history.append(total_support_hrs)

        if not rows:
            return self._empty_dataframe()

        df = pd.DataFrame(rows)
        logger.info(
            "build_features: %d rows across %d programs",
            len(df),
            df["program_id"].nunique(),
        )
        return df

    # ------------------------------------------------------------------
    # Public: single-row prediction features
    # ------------------------------------------------------------------

    async def build_features_for_prediction(
        self,
        session: AsyncSession,
        program_id: str,
        period_id: str,
    ) -> dict:
        """Compute features for one program/period for live inference.

        Uses the most recent actuals to derive lag and rolling values.
        Returns a dict with keys matching ``FEATURE_NAMES``.
        """
        programs = await self._load_programs(session)
        periods = await self._load_periods(session)
        staff_fte = await self._load_staff_fte(session, periods)

        program = programs.get(program_id)
        period = periods.get(period_id)
        if program is None or period is None:
            raise ValueError(
                f"Program {program_id!r} or period {period_id!r} not found."
            )

        # Load only this program's actuals
        actuals = await self._load_actuals(session, program_id=program_id)
        agg = self._aggregate_actuals(actuals)
        period_order = self._build_period_order(periods)

        prog_data = agg.get(program_id, {})

        # Sort prior periods chronologically
        target_idx = period_order.get(period_id, 0)
        prior_period_ids = sorted(
            [pid for pid in prog_data if period_order.get(pid, 0) < target_idx],
            key=lambda pid: period_order.get(pid, 0),
        )

        # Gather support-hours history from prior periods
        support_history: list[float] = []
        for pid in prior_period_ids:
            pd_data = prog_data[pid]
            support_history.append(
                pd_data.get((_BAM, "hrs"), 0.0) + pd_data.get((_ENG, "hrs"), 0.0)
            )

        # Current period data (may exist if actuals already loaded)
        current = prog_data.get(period_id, {})
        bam_hrs = current.get((_BAM, "hrs"), 0.0)
        eng_hrs = current.get((_ENG, "hrs"), 0.0)
        mfg_hrs = current.get((_MFG, "hrs"), 0.0)
        bam_hc = current.get((_BAM, "hc"), 0)
        eng_hc = current.get((_ENG, "hc"), 0)

        total_support_hc = bam_hc + eng_hc

        # Lags
        n = len(support_history)
        sup_lag1 = support_history[-1] if n >= 1 else 0.0
        sup_lag2 = support_history[-2] if n >= 2 else 0.0

        # Rolling 3
        if n >= 3:
            sup_rolling3 = sum(support_history[-3:]) / 3.0
        elif n > 0:
            sup_rolling3 = sum(support_history) / float(n)
        else:
            sup_rolling3 = bam_hrs + eng_hrs

        # Derived
        hrs_per_eng = _safe_div(eng_hrs, eng_hc)
        hrs_per_bam = _safe_div(bam_hrs, bam_hc)
        total_pool_hrs = bam_hrs + eng_hrs
        pct_bam = _safe_div(bam_hrs, total_pool_hrs)
        pct_eng = _safe_div(eng_hrs, total_pool_hrs)

        workdays = period.workdays or 22
        available_hrs = total_support_hc * workdays * 8
        utilization = _safe_div(bam_hrs + eng_hrs, available_hrs)

        quarter = period.quarter
        month = period.start_date.month

        pop_start = program.pop_start
        if pop_start is None and prior_period_ids:
            first_p = periods.get(prior_period_ids[0])
            pop_start = first_p.start_date if first_p else period.start_date
        elif pop_start is None:
            pop_start = period.start_date
        program_tenure_months = max(_months_between(pop_start, period.start_date), 0)

        fte_allocation = staff_fte.get((program_id, period_id), 0.0)

        # is_transition
        program_bounds = self._compute_program_bounds(agg, period_order)
        bounds = program_bounds.get(program_id)
        if bounds is not None:
            first_idx, last_idx = bounds
            current_idx = period_order.get(period_id, 0)
            is_transition = 1 if (
                current_idx <= first_idx + 2 or current_idx >= last_idx - 2
            ) else 0
        else:
            is_transition = 1

        return {
            "sup_rolling3": round(sup_rolling3, 2),
            "total_support_hc": total_support_hc,
            "hrs_per_eng": round(hrs_per_eng, 2),
            "hrs_per_bam": round(hrs_per_bam, 2),
            "bam_hc": bam_hc,
            "eng_hc": eng_hc,
            "bam_hrs": round(bam_hrs, 1),
            "eng_hrs": round(eng_hrs, 1),
            "mfg_hrs": round(mfg_hrs, 1),
            "sup_lag1": round(sup_lag1, 1),
            "sup_lag2": round(sup_lag2, 1),
            "utilization": round(utilization, 4),
            "quarter": quarter,
            "month": month,
            "program_tenure_months": program_tenure_months,
            "fte_allocation": round(fte_allocation, 2),
            "pct_bam": round(pct_bam, 4),
            "pct_eng": round(pct_eng, 4),
            "is_transition": is_transition,
        }

    # ------------------------------------------------------------------
    # Data loading helpers (async)
    # ------------------------------------------------------------------

    async def _load_programs(self, session: AsyncSession) -> dict[str, Program]:
        """Return {program.id: Program} for all programs."""
        result = await session.execute(select(Program))
        return {p.id: p for p in result.scalars().all()}

    async def _load_periods(self, session: AsyncSession) -> dict[str, ForecastPeriod]:
        """Return {period.id: ForecastPeriod} for all periods."""
        result = await session.execute(select(ForecastPeriod))
        return {p.id: p for p in result.scalars().all()}

    async def _load_actuals(
        self,
        session: AsyncSession,
        program_id: str | None = None,
    ) -> list[ActualHours]:
        """Load ActualHours rows, optionally filtered by program."""
        stmt = select(ActualHours)
        if program_id is not None:
            stmt = stmt.where(ActualHours.program_id == program_id)
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def _load_staff_fte(
        self,
        session: AsyncSession,
        periods: dict[str, ForecastPeriod],
    ) -> dict[tuple[str, str], float]:
        """Compute total FTE allocation per (program_id, period_id).

        A ProgramStaff record is active for a period if:
          staff.effective_date <= period.end_date
          AND (staff.end_date IS NULL OR staff.end_date >= period.start_date)

        FTE = sum of allocation_pct / 100 across all active BAM+ENG staff.
        """
        result = await session.execute(
            select(ProgramStaff).where(
                ProgramStaff.cost_pool.in_([_BAM, _ENG])
            )
        )
        staff_rows = list(result.scalars().all())

        fte_map: dict[tuple[str, str], float] = {}
        for period_id, period in periods.items():
            for staff in staff_rows:
                if staff.program_id != staff.program_id:
                    # always true — placeholder; real filter below
                    pass

                # Check if staff record is active during this period
                if staff.effective_date > period.end_date:
                    continue
                if staff.end_date is not None and staff.end_date < period.start_date:
                    continue

                key = (staff.program_id, period_id)
                alloc = float(staff.allocation_pct) / 100.0
                fte_map[key] = fte_map.get(key, 0.0) + alloc

        return fte_map

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def _build_period_order(
        self, periods: dict[str, ForecastPeriod]
    ) -> dict[str, int]:
        """Return {period.id: sort_index} ordered by start_date."""
        sorted_periods = sorted(periods.values(), key=lambda p: p.start_date)
        return {p.id: idx for idx, p in enumerate(sorted_periods)}

    def _aggregate_actuals(
        self, actuals: list[ActualHours]
    ) -> dict[str, dict[str, dict]]:
        """Pivot actuals into nested dicts.

        Returns::

            {
                program_id: {
                    period_id: {
                        ("BAMTL", "hrs"): float,
                        ("BAMTL", "hc"):  int,
                        ("ENGTL", "hrs"): float,
                        ("ENGTL", "hc"):  int,
                        ("MFGTL", "hrs"): float,
                        ("MFGTL", "hc"):  int,
                    },
                    ...
                },
                ...
            }

        Hours are summed across all activity types within the same cost pool.
        Headcount is maxed (not summed) across activity types — a person
        working PROD and REWORK in the same pool is still one person.
        """
        agg: dict[str, dict[str, dict]] = {}

        for ah in actuals:
            prog = agg.setdefault(ah.program_id, {})
            period_data = prog.setdefault(ah.period_id, {})

            hrs_key = (ah.cost_pool, "hrs")
            hc_key = (ah.cost_pool, "hc")

            period_data[hrs_key] = period_data.get(hrs_key, 0.0) + float(ah.total_hours)
            # Use max for headcount — multiple activity types for the same
            # cost pool shouldn't double-count people.
            period_data[hc_key] = max(
                period_data.get(hc_key, 0),
                ah.headcount,
            )

        return agg

    def _compute_program_bounds(
        self,
        agg: dict[str, dict[str, dict]],
        period_order: dict[str, int],
    ) -> dict[str, tuple[int, int]]:
        """Return {program_id: (first_period_index, last_period_index)}."""
        bounds: dict[str, tuple[int, int]] = {}
        for prog_id, period_map in agg.items():
            indices = [
                period_order[pid]
                for pid in period_map
                if pid in period_order
            ]
            if indices:
                bounds[prog_id] = (min(indices), max(indices))
        return bounds

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _empty_dataframe(self) -> pd.DataFrame:
        """Return an empty DataFrame with the correct column schema."""
        cols = ["program_id", "period_id", "label"] + FEATURE_NAMES + ["total_support_hrs"]
        return pd.DataFrame(columns=cols)
