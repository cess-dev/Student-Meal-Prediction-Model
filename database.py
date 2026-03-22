"""
database.py
===========
All database operations for the meal prediction system.
Replaces student_meals.csv, prediction_log.csv, and model_metadata.json
with proper PostgreSQL tables via SQLAlchemy.

Tables
------
    meal_orders       — raw order history (was student_meals.csv)
    prediction_log    — audit trail of every prediction served
    model_metadata    — versioning info for trained models

Setup
-----
    1. Add DATABASE_URL to your .env
    2. Run:  python database.py   — creates all tables
    3. Run:  python database.py --seed  — also loads synthetic data
"""

import argparse
import os
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, Float,
    Integer, String, Text, UniqueConstraint, create_engine, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

import config

logger = config.get_logger(__name__)


# ============================================================================
# ENGINE
# ============================================================================
def get_engine():
    url = os.getenv('DATABASE_URL')
    if not url:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set.\n"
            "Add it to your .env file:\n"
            "  DATABASE_URL=postgresql://user:password@host:5432/dbname"
        )
    # Render / Supabase sometimes give postgres:// — SQLAlchemy needs postgresql://
    if url.startswith('postgres://'):
        url = url.replace('postgres://', 'postgresql://', 1)

    engine = create_engine(
        url,
        pool_pre_ping=True,      # test connection before using from pool
        pool_size=5,
        max_overflow=10,
        echo=False,
    )
    return engine


# ============================================================================
# ORM MODELS
# ============================================================================
class Base(DeclarativeBase):
    pass


class MealOrder(Base):
    """One row per (week, student, day, meal_type, menu_item) combination."""
    __tablename__ = 'meal_orders'
    __table_args__ = (
        UniqueConstraint('week', 'student_id', 'day', 'meal_type', 'menu_item',
                         name='uq_meal_order'),
    )

    id         = Column(Integer, primary_key=True, autoincrement=True)
    week       = Column(Integer,  nullable=False, index=True)
    student_id = Column(Integer,  nullable=False, index=True)
    day        = Column(String(3), nullable=False)
    meal_type  = Column(String(20), nullable=False)
    menu_item  = Column(String(50), nullable=False)
    choice     = Column(Integer,  nullable=False)   # 0 or 1
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class PredictionLog(Base):
    """Append-only audit log of every prediction served."""
    __tablename__ = 'prediction_log'

    id          = Column(BigInteger, primary_key=True, autoincrement=True)
    logged_at   = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    student_id  = Column(Integer,  nullable=False, index=True)
    week        = Column(Integer,  nullable=False)
    day         = Column(String(3), nullable=False)
    meal_type   = Column(String(20), nullable=False)
    menu_item   = Column(String(50), nullable=False)
    probability = Column(Float,   nullable=False)
    choice      = Column(Integer, nullable=False)
    threshold   = Column(Float,   nullable=False)
    source      = Column(String(30), nullable=False)   # api-single / cli / batch


class ModelMetadata(Base):
    """One row per trained model version."""
    __tablename__ = 'model_metadata'

    id           = Column(Integer, primary_key=True, autoincrement=True)
    trained_at   = Column(DateTime(timezone=True), nullable=False, index=True)
    trained_on   = Column(Text,    nullable=True)
    n_rows       = Column(Integer, nullable=True)
    threshold    = Column(Float,   nullable=False)
    feature_cols = Column(Text,    nullable=False)   # JSON string
    model_hash   = Column(String(64), nullable=True)
    is_active    = Column(Boolean, default=True)     # only one active at a time


# ============================================================================
# SESSION FACTORY
# ============================================================================
_SessionFactory = None

def get_session() -> Session:
    global _SessionFactory
    if _SessionFactory is None:
        engine = get_engine()
        _SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)
    return _SessionFactory()


# ============================================================================
# SETUP — create tables
# ============================================================================
def create_tables():
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("All tables created (or already exist)")


# ============================================================================
# MEAL ORDERS
# ============================================================================
def load_orders(week: int | None = None, student_id: int | None = None) -> pd.DataFrame:
    """
    Load meal orders from the database into a DataFrame.
    Optionally filter by week and/or student_id.
    Returns the same schema as the old student_meals.csv.
    """
    with get_session() as session:
        query = session.query(MealOrder)
        if week is not None:
            query = query.filter(MealOrder.week == week)
        if student_id is not None:
            query = query.filter(MealOrder.student_id == student_id)
        rows = query.all()

    if not rows:
        return pd.DataFrame(columns=['week','student_id','day','meal_type','menu_item','choice'])

    return pd.DataFrame([{
        'week':       r.week,
        'student_id': r.student_id,
        'day':        r.day,
        'meal_type':  r.meal_type,
        'menu_item':  r.menu_item,
        'choice':     r.choice,
    } for r in rows])


def save_orders(df: pd.DataFrame, upsert: bool = True):
    """
    Save a DataFrame of meal orders to the database.
    If upsert=True, existing rows (same unique key) are updated.
    If upsert=False, existing rows are skipped.
    """
    required = ['week', 'student_id', 'day', 'meal_type', 'menu_item', 'choice']
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    inserted = 0
    skipped  = 0

    with get_session() as session:
        for _, row in df.iterrows():
            existing = session.query(MealOrder).filter_by(
                week=int(row['week']), student_id=int(row['student_id']),
                day=row['day'], meal_type=row['meal_type'], menu_item=row['menu_item'],
            ).first()

            if existing:
                if upsert:
                    existing.choice = int(row['choice'])
                    inserted += 1
                else:
                    skipped += 1
            else:
                session.add(MealOrder(
                    week=int(row['week']), student_id=int(row['student_id']),
                    day=row['day'], meal_type=row['meal_type'],
                    menu_item=row['menu_item'], choice=int(row['choice']),
                ))
                inserted += 1

        session.commit()

    logger.info(f"save_orders: {inserted} inserted/updated, {skipped} skipped")


# ============================================================================
# PREDICTION LOG
# ============================================================================
def log_prediction(student_id: int, week: int, day: str, meal_type: str,
                   menu_item: str, probability: float, choice: int,
                   threshold: float, source: str = 'api'):
    """Append one prediction to the log table."""
    with get_session() as session:
        session.add(PredictionLog(
            logged_at=datetime.now(timezone.utc),
            student_id=student_id, week=week, day=day,
            meal_type=meal_type, menu_item=menu_item,
            probability=round(probability, 6), choice=choice,
            threshold=threshold, source=source,
        ))
        session.commit()
    logger.debug(f"Logged: student={student_id} {day} {meal_type} {menu_item} p={probability:.3f}")


def log_predictions_batch(predictions: list[dict], source: str = 'batch'):
    """Insert many prediction rows in one transaction."""
    now = datetime.now(timezone.utc)
    with get_session() as session:
        session.bulk_insert_mappings(PredictionLog, [
            dict(
                logged_at=now,
                student_id=p['student_id'], week=p.get('week', 1),
                day=p['day'], meal_type=p['meal_type'], menu_item=p['menu_item'],
                probability=round(p.get('probability', 0), 6),
                choice=p.get('choice', 0), threshold=p.get('threshold', 0),
                source=source,
            ) for p in predictions
        ])
        session.commit()
    logger.info(f"Batch logged {len(predictions)} predictions ({source})")


def get_prediction_log(student_id: int | None = None,
                        limit: int = 1000) -> pd.DataFrame:
    """Fetch prediction log rows as a DataFrame for drift monitoring."""
    with get_session() as session:
        query = session.query(PredictionLog).order_by(PredictionLog.logged_at.desc())
        if student_id:
            query = query.filter(PredictionLog.student_id == student_id)
        rows = query.limit(limit).all()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame([{
        'logged_at':   r.logged_at, 'student_id': r.student_id,
        'week':        r.week,      'day':        r.day,
        'meal_type':   r.meal_type, 'menu_item':  r.menu_item,
        'probability': r.probability, 'choice':   r.choice,
        'threshold':   r.threshold,   'source':   r.source,
    } for r in rows])


# ============================================================================
# MODEL METADATA
# ============================================================================
def save_model_metadata(trained_at: datetime, trained_on: str, n_rows: int,
                         threshold: float, feature_cols: list,
                         model_hash: str = ''):
    """Save a new model version and mark all previous ones inactive."""
    import json
    with get_session() as session:
        # Deactivate previous versions
        session.query(ModelMetadata).filter(ModelMetadata.is_active == True).update(
            {'is_active': False}
        )
        session.add(ModelMetadata(
            trained_at=trained_at, trained_on=trained_on,
            n_rows=n_rows, threshold=threshold,
            feature_cols=json.dumps(feature_cols),
            model_hash=model_hash, is_active=True,
        ))
        session.commit()
    logger.info(f"Model metadata saved (trained_at={trained_at.isoformat()[:19]})")


def get_active_model_metadata() -> dict | None:
    """Return the currently active model's metadata as a dict."""
    import json
    with get_session() as session:
        row = session.query(ModelMetadata).filter(
            ModelMetadata.is_active == True
        ).order_by(ModelMetadata.trained_at.desc()).first()

    if not row:
        return None

    return {
        'trained_at':   row.trained_at.isoformat(),
        'trained_on':   row.trained_on,
        'n_rows':       row.n_rows,
        'threshold':    row.threshold,
        'feature_cols': json.loads(row.feature_cols),
        'model_hash':   row.model_hash,
    }


# ============================================================================
# DAILY MENU TABLE + FUNCTIONS
# ============================================================================
class DailyMenu(Base):
    """
    Published by cafeteria staff each morning.
    Tracks which items are available and whether they sold out.
    """
    __tablename__ = 'daily_menu'
    __table_args__ = (
        UniqueConstraint('date', 'meal_type', 'menu_item', name='uq_daily_item'),
    )

    id        = Column(Integer, primary_key=True, autoincrement=True)
    date      = Column(String(10),  nullable=False, index=True)   # YYYY-MM-DD
    day       = Column(String(3),   nullable=False)
    week      = Column(Integer,     nullable=False)
    meal_type = Column(String(20),  nullable=False)
    menu_item = Column(String(50),  nullable=False)
    available = Column(Boolean,     default=True)                  # False = sold out
    quantity  = Column(Integer,     nullable=True)                 # optional stock count
    created_at = Column(DateTime(timezone=True),
                        default=lambda: datetime.now(timezone.utc))


def publish_daily_menu(date_str: str, day: str, week: int,
                        meal_type: str, items: list):
    """
    Publish available items for a meal on a specific date.
    Safe to call multiple times — existing rows are updated, not duplicated.
    """
    with get_session() as session:
        for item in items:
            existing = session.query(DailyMenu).filter_by(
                date=date_str, meal_type=meal_type, menu_item=item
            ).first()
            if existing:
                existing.available = True   # re-enable if previously sold out
                existing.day       = day
                existing.week      = week
            else:
                session.add(DailyMenu(
                    date=date_str, day=day, week=week,
                    meal_type=meal_type, menu_item=item, available=True,
                ))
        session.commit()
    logger.info(f"Daily menu published: {date_str} {meal_type} — {len(items)} items")


def mark_item_sold_out(date_str: str, meal_type: str, menu_item: str):
    """Mark a specific item as sold out for a given date and meal."""
    with get_session() as session:
        row = session.query(DailyMenu).filter_by(
            date=date_str, meal_type=meal_type, menu_item=menu_item
        ).first()
        if not row:
            raise ValueError(
                f"Item '{menu_item}' not found in daily menu for {date_str} {meal_type}. "
                "Publish the daily menu first via POST /menu/daily."
            )
        row.available = False
        session.commit()
    logger.info(f"Sold out: {date_str} {meal_type} — {menu_item}")


def get_available_items(date_str: str, meal_type: str) -> list:
    """
    Return items that are available (not sold out) for a meal on a given date.
    Returns empty list if no daily menu has been published — caller should
    fall back to config.MENU_ITEMS in that case.
    """
    with get_session() as session:
        rows = session.query(DailyMenu).filter_by(
            date=date_str, meal_type=meal_type, available=True
        ).all()
    return [r.menu_item for r in rows]


def get_daily_menu(date_str: str, meal_type: str) -> list:
    """
    Return all items for a meal on a date with their availability status.
    Used by GET /menu/daily to show students what's on and what's sold out.
    """
    with get_session() as session:
        rows = session.query(DailyMenu).filter_by(
            date=date_str, meal_type=meal_type
        ).all()
    return [
        {'meal_type': r.meal_type, 'menu_item': r.menu_item, 'available': r.available}
        for r in rows
    ]


# ============================================================================
# HEALTH CHECK
# ============================================================================
def ping() -> bool:
    """Return True if the database is reachable."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text('SELECT 1'))
        return True
    except Exception as e:
        logger.error(f"Database ping failed: {e}")
        return False


# ============================================================================
# STANDALONE — create tables and optionally seed with synthetic data
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', action='store_true',
                        help='Also generate and load synthetic data into the DB')
    args = parser.parse_args()

    print("Creating tables...")
    create_tables()
    print("Tables created ✓")

    if args.seed:
        import data_loader
        print("Generating synthetic data...")
        df = data_loader.generate_synthetic_data()
        print(f"Loading {len(df):,} rows into database...")
        save_orders(df, upsert=False)
        print("Seed data loaded ✓")

    print(f"\nDatabase ping: {'✓' if ping() else '✗ FAILED'}")