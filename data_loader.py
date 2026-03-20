"""
data_loader.py
==============
Loads, validates, and feature-engineers meal order data.

Run standalone to (re)generate synthetic data:
    python data_loader.py
"""

import os
import numpy as np
import pandas as pd

import config

logger = config.get_logger(__name__)

# ============================================================================
# VALIDATION CONSTANTS
# ============================================================================
VALID_VALUES = {
    'day':       set(config.DAYS),
    'meal_type': set(config.MEAL_TYPES),
    'menu_item': {item for items in config.MENU_ITEMS.values() for item in items},
    'choice':    {0, 1},
}


# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================
def generate_synthetic_data(
    num_students: int = config.NUM_STUDENTS,
    num_weeks:    int = config.NUM_WEEKS,
    seed:         int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)
    logger.info(f"Generating synthetic data: {num_students} students x {num_weeks} weeks")
    unique_items = list({item for items in config.MENU_ITEMS.values() for item in items})
    student_affinity = {}
    for sid in range(1, num_students + 1):
        scores = np.random.dirichlet(np.ones(len(unique_items)) * 0.7)
        student_affinity[sid] = dict(zip(unique_items, scores))
    records = []
    for week in range(1, num_weeks + 1):
        for sid in range(1, num_students + 1):
            affinities = student_affinity[sid]
            for day in config.DAYS:
                attended = np.random.rand() < config.DAY_ATTENDANCE[day]
                for meal_type in config.MEAL_TYPES:
                    items = config.MENU_ITEMS[meal_type]
                    ate   = attended and np.random.rand() < config.MEAL_UPTAKE[meal_type]
                    if ate:
                        weights = np.array([affinities[i] for i in items], dtype=float)
                        weights /= weights.sum()
                        chosen  = np.random.choice(items, p=weights)
                    else:
                        chosen = None
                    for item in items:
                        records.append({
                            'week': week, 'student_id': sid, 'day': day,
                            'meal_type': meal_type, 'menu_item': item,
                            'choice': 1 if item == chosen else 0,
                        })
    df = pd.DataFrame(records)[['week','student_id','day','meal_type','menu_item','choice']]
    logger.info(f"Generated {len(df):,} rows  (chosen: {df['choice'].sum():,})")
    return df


# ============================================================================
# VALIDATION
# ============================================================================
def validate(df: pd.DataFrame) -> pd.DataFrame:
    required = ['student_id', 'day', 'meal_type', 'menu_item', 'choice']
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if 'week' not in df.columns:
        df = df.copy()
        df['week'] = 1
        logger.warning("'week' column absent — defaulting to 1")
    df['student_id'] = pd.to_numeric(df['student_id'], errors='coerce').astype('Int64')
    df['week']       = pd.to_numeric(df['week'],       errors='coerce').astype('Int64')
    df['choice']     = pd.to_numeric(df['choice'],     errors='coerce').astype('Int64')
    nulls = df[required].isnull().sum()
    if nulls.any():
        raise ValueError(f"Null values detected:\n{nulls[nulls > 0]}")
    for col, allowed in VALID_VALUES.items():
        bad = set(df[col].unique()) - allowed
        if bad:
            raise ValueError(f"Unexpected values in '{col}': {bad}")
    dup_cols = ['week', 'student_id', 'day', 'meal_type', 'menu_item']
    dups = df.duplicated(subset=dup_cols).sum()
    if dups:
        logger.warning(f"{dups} duplicate rows found — dropping")
        df = df.drop_duplicates(subset=dup_cols)
    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Compute day_num FIRST so all sorting is chronological, not alphabetical
    df['day_num'] = df['day'].map(config.DAY_ORDER)

    # Sort chronologically before computing any cumulative features
    df = df.sort_values(
        ['week', 'student_id', 'day_num', 'meal_type', 'menu_item']
    ).reset_index(drop=True)

    # 1. Historical frequency — shift(1) prevents data leakage
    df['hist_freq'] = (
        df.groupby(['student_id', 'menu_item'])['choice']
        .transform(lambda x: x.shift(1).fillna(0).cumsum())
        .astype(int)
    )

    # 2. Item popularity (global per meal_type)
    pop = (
        df.groupby(['meal_type', 'menu_item'])['choice']
        .mean().reset_index()
        .rename(columns={'choice': 'item_popularity'})
    )
    df = df.merge(pop, on=['meal_type', 'menu_item'], how='left')

    # 3. Recency — did student choose this item the previous day?
    df_sorted = df.sort_values(
        ['student_id', 'week', 'day_num', 'meal_type', 'menu_item']
    )
    df['recency'] = (
        df_sorted.groupby(['student_id', 'meal_type', 'menu_item'])['choice']
        .shift(1).fillna(0).astype(int).values
    )

    # 4. Meal streak — consecutive days this week the student ate this meal_type
    ate_meal = (
        df.groupby(['week', 'student_id', 'day', 'meal_type'])['choice']
        .max().reset_index().rename(columns={'choice': 'ate'})
    )
    ate_meal['day_num'] = ate_meal['day'].map(config.DAY_ORDER)
    ate_meal = ate_meal.sort_values(['week', 'student_id', 'meal_type', 'day_num'])
    ate_meal['meal_streak'] = (
        ate_meal.groupby(['week', 'student_id', 'meal_type'])['ate']
        .transform(lambda x: x.shift(1).fillna(0).cumsum())
        .astype(int)
    )
    df = df.merge(
        ate_meal[['week', 'student_id', 'day', 'meal_type', 'meal_streak']],
        on=['week', 'student_id', 'day', 'meal_type'], how='left',
    )

    logger.debug(f"Feature engineering complete. Shape: {df.shape}")
    return df

# ============================================================================
# PUBLIC ENTRY POINT
# ============================================================================
def load(csv_path=None) -> pd.DataFrame:
    """
    Load meal orders, validate, and feature-engineer.
    Uses the database if DATABASE_URL is set, otherwise falls back to CSV.
    """
    import os as _os
    if _os.getenv('DATABASE_URL') and csv_path is None:
        try:
            from database import load_orders
            df_raw = load_orders()
            if df_raw.empty:
                logger.warning("Database empty — seeding with synthetic data")
                df_raw = generate_synthetic_data()
                from database import save_orders
                save_orders(df_raw, upsert=False)
                logger.info(f"Seeded {len(df_raw):,} rows into database")
            else:
                logger.info(f"Loaded {len(df_raw):,} rows from database")
            return engineer_features(validate(df_raw))
        except Exception as e:
            logger.warning(f"DB load failed, falling back to CSV: {e}")

    # CSV fallback
    path = str(csv_path or config.MEALS_CSV)
    if not os.path.exists(path):
        logger.warning(f"'{path}' not found — generating synthetic data")
        df_raw = generate_synthetic_data()
        df_raw.to_csv(path, index=False)
        logger.info(f"Saved {len(df_raw):,} rows to '{path}'")
    else:
        df_raw = pd.read_csv(path)
        logger.info(f"Loaded {len(df_raw):,} rows from '{path}'")
    return engineer_features(validate(df_raw))
