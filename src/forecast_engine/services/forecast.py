"""Core forecast service — Tier 1 (headcount) + Tier 2 (GBM + SHAP).

Combines staffing-allocation-based headcount forecasts with gradient-boosted
machine-learning predictions and attaches SHAP explainability, bootstrap
confidence intervals, and a sanity-check comparison.

Public API
----------
ForecastService
    .generate_forecast(session, program_id, period_id)  -> dict
    .generate_all_programs(session, period_id)           -> list[dict]
"""

from __future__ import annotations

import asyncio
import json
import logging
from decimal import Decimal
from pathlib import Path

import joblib
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from forecast_engine.config import settings
from forecast_engine.models.actuals import ActualHours
from forecast_engine.models.demand import StaffingAllocation
from forecast_engine.models.forecast import Forecast
from forecast_engine.models.ml import ModelVersion
from forecast_engine.models.program import ForecastPeriod, Program
from forecast_engine.services.features import FEATURE_NAMES, FeatureEngineer
from forecast_engine.services.shap_engine import ShapEngine, shap_to_json

logger = logging.getLogger(__name__)

# Cost pool constants (mirror those in features.py / demand.py)
_BAM = "BAMTL"
_ENG = "ENGTL"
_SUPPORT_POOLS = {_BAM, _ENG}

# Default monthly hours constant: 21 workdays * 8 h
_DEFAULT_MONTH_HRS = 168


# ---------------------------------------------------------------------------
# Core service
# ---------------------------------------------------------------------------


class ForecastService:
    """Generate and persist Tier-1 and Tier-2 forecasts."""

    def __init__(self) -> None:
        self._feature_engineer = FeatureEngineer()

    # ------------------------------------------------------------------
    # Public: single program+period
    # ------------------------------------------------------------------

    async def generate_forecast(
        self,
        session: AsyncSession,
        program_id: str,
        period_id: str,
    ) -> dict:
        """Generate forecasts for one program/period using all available methods.

        Returns
        -------
        dict
            ``{"headcount": Forecast|None, "gbm": Forecast|None, "sanity_check": {...}}``
        """
        # Resolve the ForecastPeriod (needed for workday-based hours calc)
        period = await session.get(ForecastPeriod, period_id)

        # --- Tier 0: rolling average (always available if actuals exist) ---
        avg_forecast = await self._tier0_rolling_avg(
            session, program_id, period_id,
        )

        # --- Tier 1: headcount ---
        hc_forecast = await self._tier1_headcount(
            session, program_id, period_id, period
        )

        # --- Tier 2: GBM ---
        gbm_forecast = await self._tier2_gbm(
            session, program_id, period_id,
        )

        # --- Sanity check ---
        hc_hrs = (
            float(hc_forecast.predicted_support_hrs)
            if hc_forecast is not None
            else None
        )
        gbm_hrs = (
            float(gbm_forecast.predicted_support_hrs)
            if gbm_forecast is not None
            else None
        )
        sanity = await self._sanity_check(
            session, program_id, period_id, hc_hrs, gbm_hrs,
        )

        return {
            "rolling_avg": avg_forecast,
            "headcount": hc_forecast,
            "gbm": gbm_forecast,
            "sanity_check": sanity,
        }

    # ------------------------------------------------------------------
    # Public: batch — all active programs
    # ------------------------------------------------------------------

    async def generate_all_programs(
        self,
        session: AsyncSession,
        period_id: str,
    ) -> list[dict]:
        """Run :meth:`generate_forecast` for every active program.

        Returns a list of result dicts (one per program).
        """
        stmt = select(Program).where(Program.status == "active")
        result = await session.execute(stmt)
        programs = result.scalars().all()

        results: list[dict] = []
        for prog in programs:
            try:
                res = await self.generate_forecast(
                    session, prog.id, period_id,
                )
                results.append({"program_id": prog.id, **res})
            except Exception:
                logger.exception(
                    "Forecast failed for program %s (%s)", prog.code, prog.id
                )
                results.append({
                    "program_id": prog.id,
                    "headcount": None,
                    "gbm": None,
                    "sanity_check": {
                        "rolling_3mo_avg": None,
                        "headcount_vs_gbm_pct": None,
                        "flag": "error",
                    },
                })

        logger.info(
            "generate_all_programs: processed %d programs for period %s",
            len(results),
            period_id,
        )
        return results

    # ------------------------------------------------------------------
    # Tier 0: rolling average forecast (actuals-only)
    # ------------------------------------------------------------------

    async def _tier0_rolling_avg(
        self,
        session: AsyncSession,
        program_id: str,
        period_id: str,
        n_months: int = 3,
    ) -> "Forecast | None":
        """Forecast from rolling average of recent actuals.

        Uses the last ``n_months`` of actual support hours (BAM + ENG)
        to compute a simple average as the next-period prediction.
        Always available as long as actuals data exists.
        """
        # Get all periods ordered by start_date
        all_periods_result = await session.execute(
            select(ForecastPeriod).order_by(ForecastPeriod.start_date)
        )
        all_periods = list(all_periods_result.scalars().all())

        # Find the target period
        target_period = await session.get(ForecastPeriod, period_id)
        if target_period is None:
            return None

        # Get the N periods preceding the target
        prior_period_ids: list[str] = []
        for p in all_periods:
            if p.start_date < target_period.start_date:
                prior_period_ids.append(p.id)
        prior_period_ids = prior_period_ids[-n_months:]

        if not prior_period_ids:
            logger.debug("No prior periods for rolling avg — skipping.")
            return None

        # Query actual support hours for those periods
        stmt = (
            select(ActualHours)
            .where(
                ActualHours.program_id == program_id,
                ActualHours.period_id.in_(prior_period_ids),
                ActualHours.cost_pool.in_(list(_SUPPORT_POOLS)),
            )
        )
        result = await session.execute(stmt)
        actuals = result.scalars().all()

        if not actuals:
            logger.debug("No actuals for rolling avg — skipping.")
            return None

        # Sum hours per period and cost pool
        period_bam: dict[str, float] = {}
        period_eng: dict[str, float] = {}
        for ah in actuals:
            hrs = float(ah.total_hours)
            if ah.cost_pool == _BAM:
                period_bam[ah.period_id] = period_bam.get(ah.period_id, 0.0) + hrs
            elif ah.cost_pool == _ENG:
                period_eng[ah.period_id] = period_eng.get(ah.period_id, 0.0) + hrs

        # Average across periods
        n_periods = len(prior_period_ids)
        avg_bam = sum(period_bam.values()) / n_periods if period_bam else 0.0
        avg_eng = sum(period_eng.values()) / n_periods if period_eng else 0.0
        avg_total = avg_bam + avg_eng

        forecast = await self._upsert_forecast(
            session,
            program_id=program_id,
            period_id=period_id,
            method="rolling_avg",
            predicted_support_hrs=avg_total,
            predicted_bam_hrs=avg_bam,
            predicted_eng_hrs=avg_eng,
        )

        logger.info(
            "Tier-0 rolling_avg: program=%s period=%s -> %.1f hrs "
            "(BAM=%.1f, ENG=%.1f, based on %d months)",
            program_id, period_id, avg_total, avg_bam, avg_eng, n_periods,
        )
        return forecast

    # ------------------------------------------------------------------
    # Tier 1: headcount forecast
    # ------------------------------------------------------------------

    async def _tier1_headcount(
        self,
        session: AsyncSession,
        program_id: str,
        period_id: str,
        period: ForecastPeriod | None,
    ) -> Forecast | None:
        """Forecast from staffing allocations (FTE * allocation% * hrs/month).

        Queries the most recent *submitted* or *accepted* StaffingAllocations
        for the program, sums across BAM and ENG cost pools, and persists
        a Forecast record with ``method='headcount'``.
        """
        stmt = (
            select(StaffingAllocation)
            .where(
                StaffingAllocation.program_id == program_id,
                StaffingAllocation.cost_pool.in_(list(_SUPPORT_POOLS)),
                StaffingAllocation.status.in_(["submitted", "accepted"]),
            )
            .order_by(StaffingAllocation.created_at.desc())
        )
        result = await session.execute(stmt)
        allocations = result.scalars().all()

        if not allocations:
            logger.debug(
                "No submitted/accepted staffing allocations for program %s — "
                "skipping headcount forecast.",
                program_id,
            )
            return None

        # Hours per month: prefer period workdays, fall back to 168
        if period is not None and period.workdays:
            hrs_per_month = period.workdays * 8
        else:
            hrs_per_month = _DEFAULT_MONTH_HRS

        bam_hrs = 0.0
        eng_hrs = 0.0

        for alloc in allocations:
            fte = float(alloc.fte_count)
            pct = float(alloc.avg_allocation_pct) / 100.0
            pool_hrs = fte * pct * hrs_per_month

            if alloc.cost_pool == _BAM:
                bam_hrs += pool_hrs
            elif alloc.cost_pool == _ENG:
                eng_hrs += pool_hrs

        total_support_hrs = bam_hrs + eng_hrs

        forecast = await self._upsert_forecast(
            session,
            program_id=program_id,
            period_id=period_id,
            method="headcount",
            predicted_support_hrs=total_support_hrs,
            predicted_bam_hrs=bam_hrs,
            predicted_eng_hrs=eng_hrs,
        )

        logger.info(
            "Tier-1 headcount: program=%s period=%s -> %.1f hrs "
            "(BAM=%.1f, ENG=%.1f)",
            program_id,
            period_id,
            total_support_hrs,
            bam_hrs,
            eng_hrs,
        )
        return forecast

    # ------------------------------------------------------------------
    # Tier 2: GBM forecast
    # ------------------------------------------------------------------

    async def _tier2_gbm(
        self,
        session: AsyncSession,
        program_id: str,
        period_id: str,
    ) -> Forecast | None:
        """Predict using the active GBM model with SHAP + bootstrap CI.

        Returns ``None`` if no active model version exists.
        """
        # 1. Find the active model version
        stmt = select(ModelVersion).where(ModelVersion.is_active.is_(True))
        result = await session.execute(stmt)
        model_version = result.scalars().first()

        if model_version is None:
            logger.debug("No active model version — skipping GBM forecast.")
            return None

        # 2. Load the .joblib artifact (constrained to model_dir)
        artifact_path = Path(model_version.artifact_path).resolve()
        model_dir = settings.model_dir.resolve()
        if not str(artifact_path).startswith(str(model_dir)):
            logger.error(
                "Model artifact path %s is outside model_dir %s — refusing to load.",
                artifact_path, model_dir,
            )
            return None
        if not artifact_path.exists():
            logger.error(
                "Model artifact not found at %s — skipping GBM forecast.",
                artifact_path,
            )
            return None

        artifact: dict = joblib.load(artifact_path)
        model = artifact["model"]

        # 3. Build feature vector for this program/period
        feature_vector = await self._feature_engineer.build_features_for_prediction(
            session, program_id, period_id,
        )

        # Build ordered numpy array matching the model's expected feature order
        feature_names = artifact.get("feature_names", FEATURE_NAMES)
        X = np.array(
            [feature_vector[name] for name in feature_names],
            dtype=np.float64,
        ).reshape(1, -1)

        # 4. Predict
        predicted_total = float(model.predict(X)[0])

        # 5. SHAP explanation
        try:
            shap_engine = ShapEngine(artifact)
            shap_result = shap_engine.explain(feature_vector)
            shap_json = shap_to_json(shap_result)
        except Exception:
            logger.warning(
                "SHAP computation failed for program %s — storing without SHAP.",
                program_id,
                exc_info=True,
            )
            shap_json = None

        # 6. Bootstrap prediction interval (CPU-heavy, run in thread)
        try:
            ci_lower, ci_upper = await asyncio.to_thread(
                self._bootstrap_interval, model, X
            )
        except Exception:
            logger.warning(
                "Bootstrap interval failed for program %s — storing without CI.",
                program_id,
                exc_info=True,
            )
            ci_lower, ci_upper = None, None

        # 7. Estimate BAM/ENG split from the feature vector ratios
        pct_bam = feature_vector.get("pct_bam", 0.5)
        pct_eng = feature_vector.get("pct_eng", 0.5)
        denom = pct_bam + pct_eng
        if denom > 0:
            predicted_bam = predicted_total * (pct_bam / denom)
            predicted_eng = predicted_total * (pct_eng / denom)
        else:
            predicted_bam = predicted_total / 2.0
            predicted_eng = predicted_total / 2.0

        # 8. Persist
        feature_json = json.dumps(
            {name: feature_vector[name] for name in feature_names},
            separators=(",", ":"),
        )

        forecast = await self._upsert_forecast(
            session,
            program_id=program_id,
            period_id=period_id,
            method="gbm",
            predicted_support_hrs=predicted_total,
            predicted_bam_hrs=predicted_bam,
            predicted_eng_hrs=predicted_eng,
            model_version_id=model_version.id,
            confidence_lower=ci_lower,
            confidence_upper=ci_upper,
            shap_values=shap_json,
            feature_vector=feature_json,
        )

        logger.info(
            "Tier-2 GBM: program=%s period=%s -> %.1f hrs "
            "[CI: %.1f – %.1f] (model %s)",
            program_id,
            period_id,
            predicted_total,
            ci_lower if ci_lower is not None else 0.0,
            ci_upper if ci_upper is not None else 0.0,
            model_version.version_tag,
        )
        return forecast

    # ------------------------------------------------------------------
    # Bootstrap prediction interval
    # ------------------------------------------------------------------

    def _bootstrap_interval(
        self,
        model,
        X: np.ndarray,
        n_bootstraps: int = 100,
        ci: float = 0.80,
    ) -> tuple[float, float]:
        """Compute a prediction interval from individual GBM tree predictions.

        For ``GradientBoostingRegressor``, each ``estimator_`` is a 2-D array
        of ``DecisionTreeRegressor`` objects.  We collect per-tree predictions
        and treat them as a bootstrap distribution, then extract percentiles
        for the requested confidence interval.

        Parameters
        ----------
        model:
            A fitted ``GradientBoostingRegressor``.
        X:
            Feature matrix of shape ``(1, n_features)``.
        n_bootstraps:
            Ignored for GBM (kept for API compatibility).  All trees are used.
        ci:
            Confidence level, e.g. 0.80 for an 80% interval.

        Returns
        -------
        tuple[float, float]
            ``(lower_bound, upper_bound)``
        """
        # Gather individual tree predictions (learning_rate-scaled stage preds)
        tree_preds = np.array(
            [est.predict(X)[0] for est in model.estimators_.flatten()]
        )

        # The GBM prediction is init_value + learning_rate * sum(tree_preds).
        # For the interval we use the variance across scaled cumulative sums.
        learning_rate = model.learning_rate
        init_val = float(model.init_.predict(X)[0])

        # Build cumulative prediction trajectory across subsets of trees
        # by resampling which trees are included (true bootstrap).
        rng = np.random.default_rng(seed=42)
        n_trees = len(tree_preds)
        bootstrap_preds: list[float] = []
        for _ in range(max(n_bootstraps, n_trees)):
            sample_idx = rng.choice(n_trees, size=n_trees, replace=True)
            pred = init_val + learning_rate * tree_preds[sample_idx].sum()
            bootstrap_preds.append(pred)

        alpha = (1.0 - ci) / 2.0
        lower = float(np.percentile(bootstrap_preds, alpha * 100))
        upper = float(np.percentile(bootstrap_preds, (1.0 - alpha) * 100))

        return round(lower, 1), round(upper, 1)

    # ------------------------------------------------------------------
    # Sanity check
    # ------------------------------------------------------------------

    async def _sanity_check(
        self,
        session: AsyncSession,
        program_id: str,
        period_id: str,
        headcount_hrs: float | None,
        gbm_hrs: float | None,
    ) -> dict:
        """Compare forecasts against recent actuals for plausibility.

        - Computes a rolling 3-month average of actual BAM+ENG hours.
        - Computes the percentage divergence between headcount and GBM.
        - Flags divergence > 20%.
        """
        # Find the target period to determine which 3 months to look back
        period = await session.get(ForecastPeriod, period_id)

        # Load all periods ordered by start_date to find the 3 preceding ones
        all_periods_result = await session.execute(
            select(ForecastPeriod).order_by(ForecastPeriod.start_date)
        )
        all_periods = list(all_periods_result.scalars().all())

        # Build a list of period IDs that precede (or include) the target
        prior_period_ids: list[str] = []
        if period is not None:
            for p in all_periods:
                if p.start_date < period.start_date:
                    prior_period_ids.append(p.id)
            # Take the last 3
            prior_period_ids = prior_period_ids[-3:]

        # Query actual support hours for those periods
        rolling_avg: float | None = None
        if prior_period_ids:
            stmt = (
                select(ActualHours)
                .where(
                    ActualHours.program_id == program_id,
                    ActualHours.period_id.in_(prior_period_ids),
                    ActualHours.cost_pool.in_(list(_SUPPORT_POOLS)),
                )
            )
            result = await session.execute(stmt)
            actuals = result.scalars().all()

            # Sum hours per period, then average across periods
            period_totals: dict[str, float] = {}
            for ah in actuals:
                period_totals[ah.period_id] = (
                    period_totals.get(ah.period_id, 0.0) + float(ah.total_hours)
                )

            if period_totals:
                rolling_avg = sum(period_totals.values()) / len(period_totals)
                rolling_avg = round(rolling_avg, 1)

        # Divergence between headcount and GBM
        divergence_pct: float | None = None
        flag: str | None = None

        if headcount_hrs is not None and gbm_hrs is not None:
            max_val = max(headcount_hrs, gbm_hrs)
            if max_val > 0:
                divergence_pct = round(
                    abs(headcount_hrs - gbm_hrs) / max_val * 100, 1
                )
                if divergence_pct > 20.0:
                    flag = "divergence_warning"

        return {
            "rolling_3mo_avg": rolling_avg,
            "headcount_vs_gbm_pct": divergence_pct,
            "flag": flag,
        }

    # ------------------------------------------------------------------
    # DB upsert helper
    # ------------------------------------------------------------------

    async def _upsert_forecast(
        self,
        session: AsyncSession,
        *,
        program_id: str,
        period_id: str,
        method: str,
        predicted_support_hrs: float,
        predicted_bam_hrs: float | None = None,
        predicted_eng_hrs: float | None = None,
        model_version_id: str | None = None,
        confidence_lower: float | None = None,
        confidence_upper: float | None = None,
        shap_values: str | None = None,
        feature_vector: str | None = None,
    ) -> Forecast:
        """Create or update a Forecast row by (program_id, period_id, method).

        Uses the unique constraint ``uq_forecast_prog_period_method`` to
        decide whether to insert or update.
        """
        stmt = select(Forecast).where(
            Forecast.program_id == program_id,
            Forecast.period_id == period_id,
            Forecast.method == method,
        )
        result = await session.execute(stmt)
        existing: Forecast | None = result.scalars().first()

        if existing is not None:
            # Update in place
            existing.predicted_support_hrs = Decimal(str(round(predicted_support_hrs, 1)))
            existing.predicted_bam_hrs = (
                Decimal(str(round(predicted_bam_hrs, 1)))
                if predicted_bam_hrs is not None
                else None
            )
            existing.predicted_eng_hrs = (
                Decimal(str(round(predicted_eng_hrs, 1)))
                if predicted_eng_hrs is not None
                else None
            )
            existing.model_version_id = model_version_id
            existing.confidence_lower = (
                Decimal(str(round(confidence_lower, 1)))
                if confidence_lower is not None
                else None
            )
            existing.confidence_upper = (
                Decimal(str(round(confidence_upper, 1)))
                if confidence_upper is not None
                else None
            )
            existing.shap_values = shap_values
            existing.feature_vector = feature_vector
            await session.flush()
            return existing

        # Insert new
        forecast = Forecast(
            program_id=program_id,
            period_id=period_id,
            method=method,
            model_version_id=model_version_id,
            predicted_support_hrs=Decimal(str(round(predicted_support_hrs, 1))),
            predicted_bam_hrs=(
                Decimal(str(round(predicted_bam_hrs, 1)))
                if predicted_bam_hrs is not None
                else None
            ),
            predicted_eng_hrs=(
                Decimal(str(round(predicted_eng_hrs, 1)))
                if predicted_eng_hrs is not None
                else None
            ),
            confidence_lower=(
                Decimal(str(round(confidence_lower, 1)))
                if confidence_lower is not None
                else None
            ),
            confidence_upper=(
                Decimal(str(round(confidence_upper, 1)))
                if confidence_upper is not None
                else None
            ),
            shap_values=shap_values,
            feature_vector=feature_vector,
        )
        session.add(forecast)
        await session.flush()
        return forecast
