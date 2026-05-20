"""ActualHours, DataImport models — Phase 2 data layer."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from forecast_engine.models.base import Base, UUIDMixin, TimestampMixin


# -- Enum-like constants (stored as VARCHAR for SQLite compat) --

COST_POOLS = ("BAMTL", "ENGTL", "MFGTL")
ACTIVITY_TYPES = ("PROD", "FAI", "NRE", "ECO", "REWORK", "QUAL", "TRANS")
DATA_SOURCES = ("unanet_mcp", "oracle_mcp", "csv_upload")
IMPORT_STATUSES = ("success", "partial", "failed")


class ActualHours(UUIDMixin, Base):
    """Monthly actuals by program, cost pool, and activity type."""

    __tablename__ = "actual_hours"
    __table_args__ = (
        UniqueConstraint(
            "program_id", "period_id", "cost_pool", "activity_type",
            name="uq_actuals_prog_period_pool_activity",
        ),
    )

    program_id: Mapped[str] = mapped_column(
        ForeignKey("programs.id"), nullable=False
    )
    period_id: Mapped[str] = mapped_column(
        ForeignKey("forecast_periods.id"), nullable=False
    )
    cost_pool: Mapped[str] = mapped_column(String(10), nullable=False)  # BAMTL, ENGTL, MFGTL
    activity_type: Mapped[str] = mapped_column(String(10), default="PROD")  # PROD, FAI, NRE, ...
    total_hours: Mapped[Decimal] = mapped_column(Numeric(10, 1), nullable=False)
    headcount: Mapped[int] = mapped_column(Integer, nullable=False)
    total_cost: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    source: Mapped[str] = mapped_column(String(20), nullable=False)  # unanet_mcp, oracle_mcp, csv_upload
    import_id: Mapped[str | None] = mapped_column(ForeignKey("data_imports.id"))
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    program = relationship("Program", backref="actual_hours")
    period = relationship("ForecastPeriod", backref="actual_hours")
    data_import = relationship("DataImport", back_populates="actual_hours_records")

    def __repr__(self) -> str:
        return f"<ActualHours {self.cost_pool} {self.activity_type} {self.total_hours}h>"


class DataImport(UUIDMixin, Base):
    """Tracks each data ingestion event for audit."""

    __tablename__ = "data_imports"

    source: Mapped[str] = mapped_column(String(20), nullable=False)
    filename: Mapped[str | None] = mapped_column(String(500))
    rows_imported: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rows_skipped: Mapped[int] = mapped_column(Integer, default=0)
    periods_covered: Mapped[str | None] = mapped_column(Text)  # JSON string: ["JUN-26", "JUL-26"]
    programs_covered: Mapped[str | None] = mapped_column(Text)  # JSON string: ["531335", "C48178"]
    imported_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"))
    imported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="success")
    error_log: Mapped[str | None] = mapped_column(Text)

    # Relationships
    actual_hours_records = relationship("ActualHours", back_populates="data_import")
    imported_by_user = relationship("User", backref="data_imports")

    def __repr__(self) -> str:
        return f"<DataImport {self.source} {self.status} ({self.rows_imported} rows)>"
