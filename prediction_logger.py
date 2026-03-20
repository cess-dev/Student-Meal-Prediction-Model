"""
prediction_logger.py
====================
Logs every prediction. Uses the database if DATABASE_URL is set,
falls back to CSV if not (useful for local dev without a DB).
"""

import csv
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import config

logger = config.get_logger(__name__)
_lock  = threading.Lock()
_HEADERS = [
    'logged_at', 'student_id', 'week', 'day', 'meal_type',
    'menu_item', 'probability', 'choice', 'threshold', 'source',
]

def _use_db() -> bool:
    return bool(os.getenv('DATABASE_URL'))


def log_prediction(student_id: int, week: int, day: str, meal_type: str,
                   menu_item: str, probability: float, choice: int,
                   threshold: float, source: str = 'api'):
    if _use_db():
        try:
            from database import log_prediction as db_log
            db_log(student_id=student_id, week=week, day=day, meal_type=meal_type,
                   menu_item=menu_item, probability=probability, choice=choice,
                   threshold=threshold, source=source)
            return
        except Exception as e:
            logger.warning(f"DB log failed, falling back to CSV: {e}")

    # CSV fallback
    _csv_log({
        'logged_at': datetime.now(timezone.utc).isoformat(),
        'student_id': student_id, 'week': week, 'day': day,
        'meal_type': meal_type, 'menu_item': menu_item,
        'probability': round(probability, 6), 'choice': choice,
        'threshold': threshold, 'source': source,
    })


def log_batch(predictions: list[dict], source: str = 'batch'):
    if _use_db():
        try:
            from database import log_predictions_batch
            log_predictions_batch(predictions, source=source)
            return
        except Exception as e:
            logger.warning(f"DB batch log failed, falling back to CSV: {e}")

    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        _ensure_csv()
        with open(config.PREDICTIONS_LOG, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=_HEADERS)
            for p in predictions:
                writer.writerow({
                    'logged_at': now, 'student_id': p.get('student_id'),
                    'week': p.get('week', 1), 'day': p.get('day'),
                    'meal_type': p.get('meal_type'), 'menu_item': p.get('menu_item'),
                    'probability': round(p.get('probability', 0), 6),
                    'choice': p.get('choice', 0), 'threshold': p.get('threshold', 0),
                    'source': source,
                })


def _ensure_csv():
    path = config.PREDICTIONS_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=_HEADERS).writeheader()


def _csv_log(row: dict):
    with _lock:
        _ensure_csv()
        with open(config.PREDICTIONS_LOG, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=_HEADERS).writerow(row)