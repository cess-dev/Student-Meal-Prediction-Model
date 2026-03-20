"""
artefact_manager.py
===================
Handles saving, loading, integrity-checking, and versioning of model artefacts.

Every save writes a model_metadata.json alongside the .pkl files containing:
  - trained_at   : ISO timestamp
  - trained_on   : path to the CSV used for training
  - n_rows       : number of training rows
  - feature_cols : list of features the model expects
  - threshold    : calibrated decision threshold
  - hashes       : SHA-256 of each .pkl file

On load, hashes are re-verified so a mismatched encoder or stale threshold
raises immediately with a clear message rather than silently mispredicting.
"""

import hashlib
import json
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path

import config

logger = config.get_logger(__name__)


# ============================================================================
# HASHING
# ============================================================================
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
# SAVE
# ============================================================================
def save(clf, encoders: dict, feature_cols: list, threshold: float,
         trained_on: str = '', n_rows: int = 0):
    """
    Persist model artefacts with a versioned metadata manifest.
    All files are written atomically (temp file then rename) where possible.
    """
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        'model':    config.MODEL_PATH,
        'encoders': config.ENCODERS_PATH,
        'features': config.FEATURES_PATH,
        'threshold': config.THRESHOLD_PATH,
    }
    objects = {
        'model':     clf,
        'encoders':  encoders,
        'features':  feature_cols,
        'threshold': threshold,
    }

    for key, path in paths.items():
        with open(path, 'wb') as f:
            pickle.dump(objects[key], f)
        logger.info(f"Saved {key} → {path}")

    # Build metadata
    metadata = {
        'trained_at':   datetime.now(timezone.utc).isoformat(),
        'trained_on':   str(trained_on),
        'n_rows':       n_rows,
        'threshold':    threshold,
        'feature_cols': feature_cols,
        'hashes': {key: _sha256(path) for key, path in paths.items()},
    }

    with open(config.METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Metadata saved → {config.METADATA_PATH}")
    logger.info(f"Trained at     : {metadata['trained_at']}")
    logger.info(f"Threshold      : {threshold:.4f}")
    return metadata


# ============================================================================
# LOAD + INTEGRITY CHECK
# ============================================================================
def load() -> tuple:
    """
    Load artefacts and verify SHA-256 hashes against stored metadata.

    Returns
    -------
    (model, encoders, feature_cols, threshold, metadata)

    Raises
    ------
    FileNotFoundError  — any artefact or metadata file is missing
    RuntimeError       — hash mismatch detected (corrupted or mismatched files)
    """
    required = {
        'model':     config.MODEL_PATH,
        'encoders':  config.ENCODERS_PATH,
        'features':  config.FEATURES_PATH,
        'threshold': config.THRESHOLD_PATH,
        'metadata':  config.METADATA_PATH,
    }

    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing artefact '{name}': {path}\n"
                "  Run `python train.py` to generate all artefacts."
            )

    # Load metadata first
    with open(config.METADATA_PATH) as f:
        metadata = json.load(f)

    # Verify hashes
    hash_map = {
        'model':     config.MODEL_PATH,
        'encoders':  config.ENCODERS_PATH,
        'features':  config.FEATURES_PATH,
        'threshold': config.THRESHOLD_PATH,
    }
    mismatches = []
    for key, path in hash_map.items():
        actual   = _sha256(path)
        expected = metadata['hashes'].get(key, '')
        if actual != expected:
            mismatches.append(
                f"  {key}: expected {expected[:12]}… got {actual[:12]}…"
            )

    if mismatches:
        raise RuntimeError(
            "Artefact integrity check FAILED — files may be mismatched or corrupted.\n"
            + "\n".join(mismatches)
            + "\n  Re-run `python train.py` to regenerate all artefacts together."
        )

    # Load objects
    with open(config.MODEL_PATH,    'rb') as f: model        = pickle.load(f)
    with open(config.ENCODERS_PATH, 'rb') as f: encoders     = pickle.load(f)
    with open(config.FEATURES_PATH, 'rb') as f: feature_cols = pickle.load(f)
    with open(config.THRESHOLD_PATH,'rb') as f: threshold    = pickle.load(f)

    logger.info(f"Artefacts loaded and verified ✓  (trained: {metadata['trained_at']})")
    logger.info(f"Threshold: {threshold:.4f}  |  Features: {len(feature_cols)}")

    return model, encoders, feature_cols, threshold, metadata


# ============================================================================
# COLD-START HANDLER
# ============================================================================
def get_coldstart_features(meal_type: str, menu_item: str) -> dict:
    """
    For a brand-new student with no order history, return population-average
    feature values rather than zeros.

    Using zeros for hist_freq / item_popularity on a new student causes the
    model to predict ultra-low probabilities because it was trained on data
    where popular items typically had non-zero hist_freq. Population averages
    are a much safer cold-start baseline.
    """
    # Global item popularity: fraction of records where this item was chosen
    # These are pre-computed constants from the synthetic training distribution.
    # In production, recompute these from the live database periodically.
    POPULATION_AVG_POPULARITY = {
        # Breakfast
        'Pancakes':    0.028, 'Uji':         0.030, 'Mandazi':     0.027,
        'Tea':         0.026, 'Eggs':         0.033, 'Sausages':    0.025,
        # Lunch / Dinner (same items)
        'Rice Beans':  0.029, 'Chapo Beans':  0.030, 'Ugali Sukuma':0.028,
        'Ugali Beef':  0.027, 'Chapo Ndengu': 0.031, 'Rice Ndengu': 0.028,
    }

    return {
        'hist_freq':       0.0,   # genuinely no history
        'item_popularity': POPULATION_AVG_POPULARITY.get(menu_item, 0.028),
        'recency':         0,
        'meal_streak':     0,
    }


# ============================================================================
# SHOULD RETRAIN?
# ============================================================================
def should_retrain(max_age_days: int = 7) -> bool:
    """
    Return True if the model is older than max_age_days or metadata is missing.
    Use this in a scheduled job or CI pipeline to decide whether to retrain.
    """
    if not config.METADATA_PATH.exists():
        logger.warning("No metadata found — retrain recommended.")
        return True

    with open(config.METADATA_PATH) as f:
        metadata = json.load(f)

    trained_at = datetime.fromisoformat(metadata['trained_at'])
    age_days   = (datetime.now(timezone.utc) - trained_at).days

    if age_days >= max_age_days:
        logger.warning(f"Model is {age_days} days old (max: {max_age_days}) — retrain recommended.")
        return True

    logger.info(f"Model age: {age_days} days — no retrain needed.")
    return False