"""
predict.py
==========
Serve meal predictions from saved model artefacts.

Modes
-----
  Single:    python predict.py --student 1 --day Mon --meal Breakfast --item Pancakes
  Ranked:    python predict.py --student 5 --day Fri --meal Lunch
  Forecast:  python predict.py --student 1 --forecast --week 2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import artefact_manager
import config
import data_loader
import prediction_logger

logger = config.get_logger(__name__)


# ============================================================================
# INPUT VALIDATION
# ============================================================================
def validate_inputs(encoders: dict, day: str, meal_type: str, menu_item: str):
    checks = [
        ('day',       day,       config.DAYS),
        ('meal_type', meal_type, config.MEAL_TYPES),
        ('menu_item', menu_item, list(encoders['menu_item'].classes_)),
    ]
    errors = []
    for col, val, valid in checks:
        if val not in valid:
            errors.append(f"  '{col}' = '{val}' not recognised. Valid: {valid}")
    if errors:
        raise ValueError("Input validation failed:\n" + "\n".join(errors))


# ============================================================================
# FEATURE CONTEXT FROM CSV
# ============================================================================
def load_student_context(csv_path, student_id: int, week: int) -> pd.DataFrame:
    """Load feature-engineered rows for a student+week from CSV."""
    path = str(csv_path or config.MEALS_CSV)
    if not Path(path).exists():
        return pd.DataFrame()
    try:
        df   = data_loader.load(path)
        mask = (df['student_id'] == student_id) & (df['week'] == week)
        return df[mask].copy()
    except Exception as e:
        logger.warning(f"Could not load feature context: {e}")
        return pd.DataFrame()


def get_feature_context(ctx: pd.DataFrame, day: str, meal_type: str, menu_item: str) -> dict:
    """Extract feature values for one row; fall back to cold-start if not found."""
    if ctx.empty:
        return artefact_manager.get_coldstart_features(meal_type, menu_item)
    mask = (ctx['day']==day) & (ctx['meal_type']==meal_type) & (ctx['menu_item']==menu_item)
    rows = ctx[mask]
    if rows.empty:
        return artefact_manager.get_coldstart_features(meal_type, menu_item)
    row = rows.iloc[0]
    return {
        'hist_freq':       float(row.get('hist_freq', 0)),
        'item_popularity': float(row.get('item_popularity', 0)),
        'recency':         int(row.get('recency', 0)),
        'meal_streak':     int(row.get('meal_streak', 0)),
    }


# ============================================================================
# INFERENCE ROW BUILDER
# ============================================================================
def build_inference_row(encoders, feature_cols, student_id, day, meal_type,
                         menu_item, week=1, hist_freq=0.0, item_popularity=0.0,
                         recency=0, meal_streak=0) -> pd.DataFrame:
    row = {
        'student_id':      student_id,
        'week':            week,
        'day_enc':         encoders['day'].transform([day])[0],
        'meal_type_enc':   encoders['meal_type'].transform([meal_type])[0],
        'menu_item_enc':   encoders['menu_item'].transform([menu_item])[0],
        'hist_freq':       hist_freq,
        'item_popularity': item_popularity,
        'recency':         recency,
        'meal_streak':     meal_streak,
        'day_num':         config.DAY_ORDER[day],
    }
    return pd.DataFrame([row])[feature_cols]


# ============================================================================
# SINGLE PREDICTION
# ============================================================================
def predict_single(model, encoders, feature_cols, student_id, day, meal_type,
                   menu_item, week=1, hist_freq=0, item_popularity=0,
                   recency=0, meal_streak=0, threshold=0.15) -> dict:
    validate_inputs(encoders, day, meal_type, menu_item)
    X           = build_inference_row(encoders, feature_cols, student_id, day,
                                      meal_type, menu_item, week, hist_freq,
                                      item_popularity, recency, meal_streak)
    probability = float(model.predict_proba(X)[0][1])
    choice      = 1 if probability >= threshold else 0
    return {
        'student_id':  student_id, 'day': day, 'meal_type': meal_type,
        'menu_item':   menu_item,  'week': week,
        'choice':      choice,
        'probability': round(probability, 4),
        'threshold':   threshold,
        'label':       '✓ Will order' if choice == 1 else '✗ Won\'t order',
    }


# ============================================================================
# DIVERSITY CONSTRAINT
# ============================================================================
def apply_diversity(ranked_items: list[dict], already_predicted: set,
                    penalty: float = 0.5) -> list[dict]:
    """
    Suppress items already predicted earlier this week for the same meal_type
    by applying a probability penalty, then re-sort.

    This prevents the model from predicting the same item every day of the week.
    penalty=0.5 means a repeated item's probability is halved before re-ranking.
    """
    for item in ranked_items:
        if item['menu_item'] in already_predicted:
            item['probability'] = round(item['probability'] * penalty, 4)
            item['label']       = item['label'].replace('✓', '↩')  # visual marker
    ranked_items.sort(key=lambda x: x['probability'], reverse=True)
    return ranked_items


# ============================================================================
# RANKED MENU FOR ONE MEAL SITTING
# ============================================================================
def predict_meal_options(model, encoders, feature_cols, student_id, day,
                          meal_type, week=1, threshold=0.15,
                          csv_path=None, already_predicted=None,
                          available_items=None) -> pd.DataFrame:
    """
    Predict and rank all items for a meal sitting.
    available_items: list from daily menu. Falls back to full config menu if None.
    already_predicted: set of item names chosen earlier this week (diversity).
    """
    ctx   = load_student_context(csv_path, student_id, week)
    items = available_items if available_items else config.MENU_ITEMS[meal_type]
    rows  = []
    for item in items:
        feat = get_feature_context(ctx, day, meal_type, item)
        rows.append(predict_single(model, encoders, feature_cols, student_id,
                                   day, meal_type, item, week,
                                   hist_freq=feat['hist_freq'],
                                   item_popularity=feat['item_popularity'],
                                   recency=feat['recency'],
                                   meal_streak=feat['meal_streak'],
                                   threshold=threshold))

    if already_predicted:
        rows = apply_diversity(rows, already_predicted)

    df = pd.DataFrame(rows).sort_values('probability', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)

    # Top item always gets a definitive label
    if df.loc[0, 'choice'] == 0:
        df.loc[0, 'label'] = '→ Best guess'

    return df[['rank', 'menu_item', 'probability', 'label']]


# ============================================================================
# WEEKLY FORECAST WITH DIVERSITY
# ============================================================================
def predict_weekly(model, encoders, feature_cols, student_id, week=1,
                   threshold=0.15, csv_path=None) -> pd.DataFrame:
    """
    Full weekly forecast. Applies diversity constraint across days so the same
    item is not predicted every day for the same meal_type.
    """
    ctx = load_student_context(csv_path, student_id, week)
    rows = []

    # Track already-predicted items per meal_type for diversity
    predicted_per_meal: dict[str, set] = {mt: set() for mt in config.MEAL_TYPES}

    for day in config.DAYS:
        for meal_type in config.MEAL_TYPES:
            items = config.MENU_ITEMS[meal_type]
            item_scores = []
            for item in items:
                feat = get_feature_context(ctx, day, meal_type, item)
                result = predict_single(
                    model, encoders, feature_cols, student_id, day, meal_type,
                    item, week, hist_freq=feat['hist_freq'],
                    item_popularity=feat['item_popularity'],
                    recency=feat['recency'], meal_streak=feat['meal_streak'],
                    threshold=threshold,
                )
                item_scores.append(result)

            # Apply diversity penalty before picking top-1
            item_scores = apply_diversity(item_scores, predicted_per_meal[meal_type])
            best = max(item_scores, key=lambda x: x['probability'])

            # Register this prediction so future days get penalised for repeats
            predicted_per_meal[meal_type].add(best['menu_item'])

            # Attendance context
            p_attend = config.DAY_ATTENDANCE.get(day, 0.9)
            p_eat    = config.MEAL_UPTAKE.get(meal_type, 0.8)
            p_show   = p_attend * p_eat
            if p_show >= 0.80:
                attendance = f"High ({p_show:.0%})"
            elif p_show >= 0.50:
                attendance = f"Moderate ({p_show:.0%})"
            else:
                attendance = f"Low ({p_show:.0%})"

            prob = best['probability']
            choice_label = f"✓ Will order ({prob:.0%})" if prob >= threshold else f"→ Best guess ({prob:.0%})"

            rows.append({
                'day':            day,
                'meal_type':      meal_type,
                'menu_item':      best['menu_item'],
                'probability':    prob,
                'predicted_choice': choice_label,
                'attendance':     attendance,
            })

            # Log prediction
            prediction_logger.log_prediction(
                student_id=student_id, week=week, day=day, meal_type=meal_type,
                menu_item=best['menu_item'], probability=prob,
                choice=best['choice'], threshold=threshold, source='cli-forecast',
            )

    df = pd.DataFrame(rows)
    df['day_num'] = df['day'].map(config.DAY_ORDER)
    df = df.sort_values(['day_num', 'meal_type']).drop(columns='day_num')
    return df[['day', 'meal_type', 'menu_item', 'probability', 'predicted_choice', 'attendance']]


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--student',  type=int, required=True)
    parser.add_argument('--day',      type=str, default=None)
    parser.add_argument('--meal',     type=str, default=None)
    parser.add_argument('--item',     type=str, default=None)
    parser.add_argument('--week',     type=int, default=2)
    parser.add_argument('--forecast', action='store_true')
    parser.add_argument('--csv',      type=str, default=None)
    args = parser.parse_args()

    try:
        model, encoders, feature_cols, threshold, meta = artefact_manager.load()
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n❌ {e}")
        sys.exit(1)

    print(f"\n  Model loaded ✓  (trained: {meta['trained_at'][:19]})")
    print(f"  Threshold: {threshold:.2f}  |  {'='*44}")

    try:
        if args.forecast:
            print(f"\n  Weekly forecast — Student {args.student}  (Week {args.week})")
            df = predict_weekly(model, encoders, feature_cols, args.student,
                                args.week, threshold, args.csv)
            print(df.to_string(index=False))

        elif args.day and args.meal and not args.item:
            print(f"\n  Ranked menu — Student {args.student} | {args.day} {args.meal}")
            df = predict_meal_options(model, encoders, feature_cols, args.student,
                                      args.day, args.meal, args.week, threshold, args.csv)
            print(df.to_string(index=False))

        elif args.day and args.meal and args.item:
            result = predict_single(model, encoders, feature_cols, args.student,
                                    args.day, args.meal, args.item, args.week,
                                    threshold=threshold)
            print(f"\n  Student {result['student_id']} | {result['day']} "
                  f"{result['meal_type']} | {result['menu_item']}")
            print(f"  Prediction  : {result['label']}")
            print(f"  Probability : {result['probability']:.1%}  (threshold: {threshold:.2f})")
            prediction_logger.log_prediction(
                student_id=args.student, week=args.week, day=args.day,
                meal_type=args.meal, menu_item=args.item,
                probability=result['probability'], choice=result['choice'],
                threshold=threshold, source='cli-single',
            )
        else:
            parser.print_help()

    except ValueError as e:
        print(f"\n❌ {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()