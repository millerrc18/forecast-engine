"""Forecast, ForecastOverride models — prediction and PM override layer."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from forecast_engine.models.base import Base, UUIDMixin


# -- Enum-like constants (stored as VARCHAR for SQLite compat) --

FORECAST_METHODS = ("rolling_avg", "headcount", "gbm", "ratio_legacy")
OVERRIDE_REASON_CODES = (
    "staffing_change",
    "scope_change",
    "schedule_shift",
    "pto_leave",
    "rework_expected",
    "milestone_moved",
    "pm_judgment",
    "other",
)


class Forecast(UUIDMixin, Base):
    """Model/method prediction for a program-period combination."""

    __tablename__ = "forecasts"
    __table_args__ = (
        UniqueConstraint(
            "program_id", "period_id", "method",
            name="uq_forecast_prog_period_method",
        ),
    )

    program_id: Mapped[str] = mapped_column(
        ForeignKey("programs.id"), nullable=False
    )
    period_id: Mapped[str] = mapped_column(
        ForeignKey("forecast_periods.id"), nullable=False
    )
    method: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # headcount, gbm, ratio_legacy
    model_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_versions.id")
    )  # NULL for headcount/ratio
    predicted_support_hrs: Mapped[Decimal] = mapped_column(
        Numeric(10, 1), nullable=False
    )
    predicted_bam_hrs: Mapped[Decimal | None] = mapped_column(Numeric(10, 1))
    predicted_eng_hrs: Mapped[Decimal | None] = mapped_column(Numeric(10, 1))
    confidence_lower: Mapped[Decimal | None] = mapped_column(Numeric(10, 1))  # 80% PI low
    confidence_upper: Mapped[Decimal | None] = mapped_column(Numeric(10, 1))  # 80% PI high
    shap_values: Mapped[str | None] = mapped_column(Text)  # JSON string of SHAP breakdown
    feature_vector: Mapped[str | None] = mapped_column(Text)  # JSON string of input features
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    program = relationship("Program", backref="forecasts")
    period = relationship("ForecastPeriod", backref="forecasts")
    model_version = relationship("ModelVersion", backref="forecasts")
    overrides = relationship("ForecastOverride", back_populates="forecast")

    def __repr__(self) -> str:
        return f"<Forecast {self.method} {self.predicted_support_hrs}h>"


class ForecastOverride(UUIDMixin, Base):
    """PM override of a model forecast with audit trail."""

    __tablename__ = "forecast_overrides"

    forecast_id: Mapped[str] = mapped_column(
        ForeignKey("forecasts.id"), nullable=False
    )
    overridden_by: Mapped[str] = mapped_column(
        ForeignKey("users.id"), nullable=False
    )
    original_hrs: Mapped[Decimal] = mapped_column(
        Numeric(10, 1), nullable=False
    )  # snapshot of model prediction
    adjusted_hrs: Mapped[Decimal] = mapped_column(
        Numeric(10, 1), nullable=False
    )  # PM's final number
    reason_code: Mapped[str] = mapped_column(
        String(30), nullable=False
    )  # staffing_change, scope_change, ...
    reason_text: Mapped[str | None] = mapped_column(Text)  # free-text explanation
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    forecast = relationship("Forecast", back_populates="overrides")
    overridden_by_user = relationship("User", backref="forecast_overrides")

    def __repr__(self) -> str:
        return f"<ForecastOverride {self.original_hrs}h -> {self.adjusted_hrs}h ({self.reason_code})>"
