"""
retrain.py
==========
Scheduled retraining pipeline.

Appends new order data to the master CSV, checks whether retraining is needed,
retrains if so, and writes new artefacts — all without downtime.

Usage
-----
  Append new data and retrain if model is stale:
      python retrain.py --new-data new_orders.csv

  Force retrain even if model is fresh:
      python retrain.py --force

  Check staleness only (dry run):
      python retrain.py --check

Schedule with cron (retrain every Monday at 2 AM):
      0 2 * * 1  cd /app && python retrain.py --new-data /data/weekly_orders.csv
"""

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import artefact_manager
import config
import data_loader

logger = config.get_logger(__name__)


# ============================================================================
# APPEND NEW DATA
# ============================================================================
def append_new_data(new_csv: str) -> int:
    """
    Validate and append new_csv rows to the master MEALS_CSV.
    Returns number of rows appended.

    The new CSV must have the same schema as the master:
        week, student_id, day, meal_type, menu_item, choice
    """
    new_path = Path(new_csv)
    if not new_path.exists():
        raise FileNotFoundError(f"New data file not found: {new_csv}")

    df_new = pd.read_csv(new_csv)
    logger.info(f"New data: {len(df_new):,} rows from '{new_csv}'")

    # Validate schema
    df_new = data_loader.validate(df_new)

    master_path = config.MEALS_CSV
    if master_path.exists():
        df_master = pd.read_csv(master_path)
        df_combined = pd.concat([df_master, df_new], ignore_index=True)

        # Drop duplicates (same week/student/day/meal/item)
        dup_cols    = ['week', 'student_id', 'day', 'meal_type', 'menu_item']
        before      = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=dup_cols, keep='last')
        dropped     = before - len(df_combined)
        if dropped:
            logger.warning(f"Dropped {dropped} duplicate rows during merge")
    else:
        df_combined = df_new
        logger.warning(f"Master CSV not found — creating from new data only")

    df_combined.to_csv(master_path, index=False)
    n_appended = len(df_new)
    logger.info(f"Master CSV updated: {len(df_combined):,} total rows  (+{n_appended} new)")
    return n_appended


# ============================================================================
# RUN TRAINING
# ============================================================================
def run_training():
    """
    Imports and runs train.main() then verifies the new artefacts load cleanly.
    """
    import train as _train   # imported here to avoid circular deps at module level
    _train.main(csv_path=str(config.MEALS_CSV))

    # Verify new artefacts load and pass integrity check
    model, encoders, feature_cols, threshold, meta = artefact_manager.load()
    logger.info(f"New artefacts verified ✓  threshold={threshold:.3f}")
    return meta


# ============================================================================
# BACKUP OLD ARTEFACTS
# ============================================================================
def backup_artefacts():
    """Copy current artefacts to a timestamped backup directory."""
    ts      = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    backup  = config.MODEL_DIR / f'backup_{ts}'
    backup.mkdir(parents=True, exist_ok=True)

    for src in [config.MODEL_PATH, config.ENCODERS_PATH,
                config.FEATURES_PATH, config.THRESHOLD_PATH, config.METADATA_PATH]:
        if src.exists():
            shutil.copy2(src, backup / src.name)

    logger.info(f"Old artefacts backed up → {backup}")
    return backup


# ============================================================================
# MAIN
# ============================================================================
def main(new_data: str | None = None, force: bool = False, check_only: bool = False):
    logger.info("=" * 50)
    logger.info("RETRAIN PIPELINE STARTED")
    logger.info("=" * 50)

    # ── 1. Staleness check ────────────────────────────────────────────────────
    stale = artefact_manager.should_retrain(max_age_days=7)

    if check_only:
        status = "STALE — retrain recommended" if stale else "FRESH — no retrain needed"
        logger.info(f"Model status: {status}")
        return

    if not stale and not force:
        logger.info("Model is fresh. Use --force to retrain anyway. Exiting.")
        return

    # ── 2. Append new data if provided ────────────────────────────────────────
    if new_data:
        try:
            n = append_new_data(new_data)
            logger.info(f"Appended {n:,} new rows to master CSV")
        except Exception as e:
            logger.error(f"Failed to append new data: {e}")
            raise

    # ── 3. Backup existing artefacts ──────────────────────────────────────────
    if config.MODEL_PATH.exists():
        backup_artefacts()

    # ── 4. Retrain ────────────────────────────────────────────────────────────
    logger.info("Starting training...")
    try:
        meta = run_training()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    logger.info("=" * 50)
    logger.info(f"RETRAIN COMPLETE  trained_at={meta['trained_at'][:19]}")
    logger.info("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrain the meal prediction model.')
    parser.add_argument('--new-data',  type=str, default=None, help='Path to new orders CSV to append')
    parser.add_argument('--force',     action='store_true',    help='Force retrain even if model is fresh')
    parser.add_argument('--check',     action='store_true',    help='Only check staleness, do not retrain')
    args = parser.parse_args()
    main(new_data=args.new_data, force=args.force, check_only=args.check)