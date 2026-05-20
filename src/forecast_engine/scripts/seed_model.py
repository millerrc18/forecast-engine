"""Seed the v1.0 forecast model from existing actuals data.

Usage::

    python -m forecast_engine.scripts.seed_model

The script will:
1. Ensure database tables exist (init_db).
2. Verify ActualHours data is present — exit with guidance if not.
3. Skip if a v1.x model already exists.
4. Train a GBM model via TrainingService and activate it.
5. Print a summary of the trained model.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from sqlalchemy import func, select

from forecast_engine.models.actuals import ActualHours
from forecast_engine.models.base import async_session, init_db
from forecast_engine.models.ml import ModelVersion
from forecast_engine.services.training import TrainingService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

VERSION_TAG = "v1.0-2026Q2"


async def main() -> None:
    """Train and activate the initial v1.0 model."""

    # 1. Ensure tables exist
    print("[seed_model] Initializing database...")
    await init_db()

    async with async_session() as session:
        # 2. Check for actuals data
        row_count = await session.scalar(
            select(func.count()).select_from(ActualHours)
        )
        if not row_count:
            print(
                "\n[seed_model] No ActualHours data found.\n"
                "  Import actuals before training a model. You can:\n"
                "    - Upload a CSV via the web UI  (/upload)\n"
                "    - Run the Unanet/Oracle MCP import scripts\n"
                "\nExiting without training."
            )
            sys.exit(1)

        print(f"[seed_model] Found {row_count:,} actuals rows.")

        # 3. Check for existing v1.x model
        existing = await session.scalar(
            select(func.count())
            .select_from(ModelVersion)
            .where(ModelVersion.version_tag.startswith("v1"))
        )
        if existing:
            print("[seed_model] Model v1 already exists, skipping.")
            return

        # 4. Train
        print(f"[seed_model] Training model {VERSION_TAG}...")
        svc = TrainingService()
        try:
            model = await svc.train_model(
                session=session,
                user_id="system",
                version_tag=VERSION_TAG,
            )
        except ValueError as exc:
            print(f"\n[seed_model] Training failed: {exc}")
            sys.exit(1)
        except Exception as exc:
            logger.exception("Unexpected error during training")
            print(f"\n[seed_model] Training failed unexpectedly: {exc}")
            sys.exit(1)

        # 5. Activate
        await svc.activate_model(session, model.id)
        await session.commit()

        # 6. Fetch overall MAPE for summary
        from forecast_engine.models.ml import ModelMetric  # noqa: E402

        mape_row = await session.scalar(
            select(ModelMetric.metric_value).where(
                ModelMetric.model_version_id == model.id,
                ModelMetric.metric_name == "mape",
                ModelMetric.program_id.is_(None),
            )
        )
        overall_mape = float(mape_row) if mape_row is not None else 0.0

        # 7. Summary
        print(
            f"\n{'=' * 50}\n"
            f"  Model trained and activated successfully!\n"
            f"{'=' * 50}\n"
            f"  Version tag:     {model.version_tag}\n"
            f"  Observations:    {model.n_observations:,}\n"
            f"  Programs:        {model.n_programs}\n"
            f"  Overall MAPE:    {overall_mape:.2f}%\n"
            f"  Artifact path:   {model.artifact_path}\n"
            f"{'=' * 50}"
        )


if __name__ == "__main__":
    asyncio.run(main())
