"""
config.py
=========
Single source of truth for all configuration.
Every other module imports from here — no hardcoded paths or magic numbers.

Environment variables (set in .env or shell):
    MODEL_DIR       directory where .pkl artefacts live  (default: current dir)
    DATA_DIR        directory where CSVs live            (default: current dir)
    LOG_LEVEL       DEBUG / INFO / WARNING               (default: INFO)
    LOG_FILE        path to log file                     (default: logs/app.log)
    API_KEY         secret key for API auth              (default: dev-key-change-me)
    NUM_STUDENTS    number of students in simulation     (default: 200)
    NUM_WEEKS       weeks of synthetic history           (default: 4)
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()   # reads .env if present; harmless if absent

# ── Directories ───────────────────────────────────────────────────────────────
MODEL_DIR = Path(os.getenv('MODEL_DIR', '.')).resolve()
DATA_DIR  = Path(os.getenv('DATA_DIR',  '.')).resolve()
LOG_DIR   = Path(os.getenv('LOG_DIR',   'logs')).resolve()

# ── Artefact paths ────────────────────────────────────────────────────────────
MODEL_PATH     = MODEL_DIR / 'meal_model.pkl'
ENCODERS_PATH  = MODEL_DIR / 'encoders.pkl'
FEATURES_PATH  = MODEL_DIR / 'feature_cols.pkl'
THRESHOLD_PATH = MODEL_DIR / 'threshold.pkl'
METADATA_PATH  = MODEL_DIR / 'model_metadata.json'   # versioning + integrity

# ── Data paths ────────────────────────────────────────────────────────────────
MEALS_CSV      = DATA_DIR / os.getenv('MEALS_CSV', 'student_meals.csv')
PREDICTIONS_LOG = DATA_DIR / 'prediction_log.csv'    # audit trail

# ── Domain constants ──────────────────────────────────────────────────────────
DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
MEAL_TYPES = ['Breakfast', 'Lunch', 'Dinner']
MENU_ITEMS = {
    'Breakfast': ['Pancakes', 'Uji', 'Mandazi', 'Tea', 'Eggs', 'Sausages'],
    'Lunch':     ['Rice Beans', 'Chapo Beans', 'Ugali Sukuma', 'Ugali Beef', 'Chapo Ndengu', 'Rice Ndengu'],
    'Dinner':    ['Rice Beans', 'Chapo Beans', 'Ugali Sukuma', 'Ugali Beef', 'Chapo Ndengu', 'Rice Ndengu'],
}
DAY_ORDER = {d: i for i, d in enumerate(DAYS)}

# ── Simulation parameters ─────────────────────────────────────────────────────
NUM_STUDENTS = int(os.getenv('NUM_STUDENTS', 200))
NUM_WEEKS    = int(os.getenv('NUM_WEEKS',    4))

# ── Day / meal attendance rates (used for attendance labels) ──────────────────
DAY_ATTENDANCE = {
    'Mon': 0.95, 'Tue': 0.95, 'Wed': 0.93,
    'Thu': 0.92, 'Fri': 0.88, 'Sat': 0.55, 'Sun': 0.40,
}
MEAL_UPTAKE = {'Breakfast': 0.65, 'Lunch': 0.92, 'Dinner': 0.80}

# ── API security ──────────────────────────────────────────────────────────────
API_KEY = os.getenv('API_KEY', 'dev-key-change-me')

# ── Model training ────────────────────────────────────────────────────────────
RF_N_ESTIMATORS  = int(os.getenv('RF_N_ESTIMATORS', 200))
RF_MAX_DEPTH     = int(os.getenv('RF_MAX_DEPTH',    12))
RF_MIN_LEAF      = int(os.getenv('RF_MIN_LEAF',     5))
CV_FOLDS         = int(os.getenv('CV_FOLDS',        5))

# ── Feature columns (single authoritative list) ───────────────────────────────
FEATURE_COLS = [
    'student_id', 'week', 'day_enc', 'meal_type_enc', 'menu_item_enc',
    'hist_freq', 'item_popularity', 'recency', 'meal_streak', 'day_num',
]
CAT_COLS = ['day', 'meal_type', 'menu_item']

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
LOG_FILE  = Path(os.getenv('LOG_FILE', str(LOG_DIR / 'app.log')))

def get_logger(name: str) -> logging.Logger:
    """
    Return a logger that writes to both console and LOG_FILE.
    Call once per module:  logger = config.get_logger(__name__)
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    logger  = logging.getLogger(name)
    if logger.handlers:          # avoid adding duplicate handlers on reimport
        return logger

    level   = getattr(logging, LOG_LEVEL, logging.INFO)
    fmt     = logging.Formatter('%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
                                 datefmt='%Y-%m-%d %H:%M:%S')

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(fmt)

    logger.setLevel(level)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger