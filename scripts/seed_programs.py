"""Seed the database with 14 programs and forecast periods."""

import asyncio
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import select
from forecast_engine.models.base import async_session, init_db
from forecast_engine.models.program import Program, ForecastPeriod, User

PROGRAMS = [
    ("531335", "GAC Elevator"),
    ("C48178", "Aeronose Radome"),
    ("537786", "F-16 DLA"),
    ("538346", "NGJ Lot 4 Arrays"),
    ("521938", "Boeing Triband"),
    ("536670", "F-15 EX Radomes"),
    ("533906", "F-18 AESA"),
    ("526074", "F-16 New Build"),
    ("521462", "RWBOA"),
    ("526068", "IFEC Radomes"),
    ("531406", "NGJ LRIP 3"),
    ("539211", "JSF Lot 18-19"),
    ("539212", "C130"),
    ("541021", "F-16 Retrofits"),
]

PERIODS = [
    # (label, start, end, workdays, quarter, fy)
    ("JAN-25", date(2025, 1, 1), date(2025, 1, 31), 22, 1, 2025),
    ("FEB-25", date(2025, 2, 1), date(2025, 2, 28), 20, 1, 2025),
    ("MAR-25", date(2025, 3, 1), date(2025, 3, 31), 21, 1, 2025),
    ("APR-25", date(2025, 4, 1), date(2025, 4, 30), 22, 2, 2025),
    ("MAY-25", date(2025, 5, 1), date(2025, 5, 31), 21, 2, 2025),
    ("JUN-25", date(2025, 6, 1), date(2025, 6, 30), 21, 2, 2025),
    ("JUL-25", date(2025, 7, 1), date(2025, 7, 31), 22, 3, 2025),
    ("AUG-25", date(2025, 8, 1), date(2025, 8, 31), 21, 3, 2025),
    ("SEP-25", date(2025, 9, 1), date(2025, 9, 30), 21, 3, 2025),
    ("OCT-25", date(2025, 10, 1), date(2025, 10, 31), 23, 4, 2025),
    ("NOV-25", date(2025, 11, 1), date(2025, 11, 30), 18, 4, 2025),
    ("DEC-25", date(2025, 12, 1), date(2025, 12, 31), 20, 4, 2025),
    ("JAN-26", date(2026, 1, 1), date(2026, 1, 31), 21, 1, 2026),
    ("FEB-26", date(2026, 2, 1), date(2026, 2, 28), 20, 1, 2026),
    ("MAR-26", date(2026, 3, 1), date(2026, 3, 31), 22, 1, 2026),
    ("APR-26", date(2026, 4, 1), date(2026, 4, 30), 22, 2, 2026),
    ("MAY-26", date(2026, 5, 1), date(2026, 5, 31), 20, 2, 2026),
    ("JUN-26", date(2026, 6, 1), date(2026, 6, 30), 22, 2, 2026),
    ("JUL-26", date(2026, 7, 1), date(2026, 7, 31), 22, 3, 2026),
    ("AUG-26", date(2026, 8, 1), date(2026, 8, 31), 21, 3, 2026),
    ("SEP-26", date(2026, 9, 1), date(2026, 9, 30), 21, 3, 2026),
    ("OCT-26", date(2026, 10, 1), date(2026, 10, 31), 22, 4, 2026),
    ("NOV-26", date(2026, 11, 1), date(2026, 11, 30), 19, 4, 2026),
    ("DEC-26", date(2026, 12, 1), date(2026, 12, 31), 21, 4, 2026),
]

DEMO_USERS = [
    ("ryan.c.miller", "Ryan Miller", "ryan.c.miller@gd-ms.com", "pm"),
    ("demo.funcmgr", "Demo Func Manager", "demo.funcmgr@gd-ms.com", "func_mgr"),
    ("demo.leadership", "Demo Director", "demo.leadership@gd-ms.com", "leadership"),
    ("admin", "System Admin", "admin@gd-ms.com", "admin"),
]


async def seed():
    await init_db()

    async with async_session() as session:
        # Seed users
        for username, display, email, role in DEMO_USERS:
            exists = await session.execute(select(User).where(User.username == username))
            if not exists.scalar_one_or_none():
                session.add(User(
                    username=username, display_name=display, email=email, role=role
                ))
        await session.flush()

        # Get PM user for assigning programs
        pm = (await session.execute(
            select(User).where(User.username == "ryan.c.miller")
        )).scalar_one()

        # Seed programs
        for code, name in PROGRAMS:
            exists = await session.execute(select(Program).where(Program.code == code))
            if not exists.scalar_one_or_none():
                pm_id = pm.id if code in ("531335", "C48178") else None
                session.add(Program(code=code, name=name, pm_user_id=pm_id))

        # Seed periods
        for label, start, end, wd, q, fy in PERIODS:
            exists = await session.execute(
                select(ForecastPeriod).where(ForecastPeriod.label == label)
            )
            if not exists.scalar_one_or_none():
                session.add(ForecastPeriod(
                    label=label, start_date=start, end_date=end,
                    workdays=wd, quarter=q, fiscal_year=fy
                ))

        await session.commit()

    print(f"Seeded {len(PROGRAMS)} programs, {len(PERIODS)} periods, {len(DEMO_USERS)} users.")


if __name__ == "__main__":
    asyncio.run(seed())
