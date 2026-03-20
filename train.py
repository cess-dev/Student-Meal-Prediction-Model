"""
train.py
========
Train a Random Forest, evaluate with K-Fold CV, calibrate threshold,
and save all artefacts via artefact_manager.

Usage:
    python train.py
    python train.py --csv student_meals.csv --folds 5 --trees 200
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

import config
import data_loader
import artefact_manager

warnings.filterwarnings('ignore')
logger = config.get_logger(__name__)


# ============================================================================
# ENCODING
# ============================================================================
def encode_features(df: pd.DataFrame):
    df_enc   = df.copy()
    encoders = {}
    for col in config.CAT_COLS:
        le = LabelEncoder()
        df_enc[f'{col}_enc'] = le.fit_transform(df_enc[col])
        encoders[col] = le
    return df_enc, encoders


# ============================================================================
# OPTIMAL THRESHOLD
# ============================================================================
def find_optimal_threshold(clf, X_val, y_val) -> float:
    y_proba    = clf.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.05, 0.51, 0.01)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        score = f1_score(y_val, (y_proba >= t).astype(int), zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, t
    logger.info(f"Optimal threshold: {best_t:.2f}  (F1={best_f1:.4f})")
    return float(round(best_t, 2))


# ============================================================================
# CROSS-VALIDATION
# ============================================================================
def cross_validate_model(clf, X, y, n_folds=5):
    skf     = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = cross_validate(
        clf, X, y, cv=skf, n_jobs=-1,
        scoring={'f1':'f1','precision':'precision','recall':'recall','roc_auc':'roc_auc'},
    )
    print(f"\n{'='*60}")
    print(f"  {n_folds}-FOLD CROSS-VALIDATION  (Positive class = Chosen)")
    print(f"{'='*60}")
    print(f"  {'Fold':<6} {'F1':>8} {'Precision':>11} {'Recall':>9} {'ROC-AUC':>10}")
    print(f"  {'-'*50}")
    for i in range(n_folds):
        print(f"  {i+1:<6} {results['test_f1'][i]:>8.4f}"
              f" {results['test_precision'][i]:>11.4f}"
              f" {results['test_recall'][i]:>9.4f}"
              f" {results['test_roc_auc'][i]:>10.4f}")
    print(f"  {'-'*50}")
    print(f"  {'Mean':<6} {results['test_f1'].mean():>8.4f}"
          f" {results['test_precision'].mean():>11.4f}"
          f" {results['test_recall'].mean():>9.4f}"
          f" {results['test_roc_auc'].mean():>10.4f}")
    print(f"  {'±Std':<6} {results['test_f1'].std():>8.4f}"
          f" {results['test_precision'].std():>11.4f}"
          f" {results['test_recall'].std():>9.4f}"
          f" {results['test_roc_auc'].std():>10.4f}")
    print(f"{'='*60}\n")
    return results


# ============================================================================
# HOLD-OUT EVALUATION
# ============================================================================
def evaluate_holdout(clf, X_test, y_test, threshold):
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)
    print(f"\n{'='*60}")
    print("  HOLD-OUT TEST SET METRICS")
    print(f"{'='*60}")
    print(f"  Threshold (calibrated) : {threshold:.2f}")
    print(f"  Accuracy  (info only)  : {accuracy_score(y_test,y_pred):.4f}")
    print(f"  F1-score  (Chosen)     : {f1_score(y_test,y_pred):.4f}  <- primary")
    print(f"  Precision (Chosen)     : {precision_score(y_test,y_pred,zero_division=0):.4f}")
    print(f"  Recall    (Chosen)     : {recall_score(y_test,y_pred,zero_division=0):.4f}")
    print(f"  ROC-AUC               : {roc_auc_score(y_test,y_proba):.4f}")
    print(f"\n{classification_report(y_test,y_pred,target_names=['Not Chosen','Chosen'])}")
    fi = pd.Series(clf.feature_importances_, index=config.FEATURE_COLS).sort_values(ascending=False)
    print("  Feature importances:")
    for feat, imp in fi.items():
        print(f"    {feat:<22} {imp:.4f}  {'█'*int(imp*60)}")
    print()


# ============================================================================
# MAIN
# ============================================================================
def main(csv_path=None, n_folds=config.CV_FOLDS, n_estimators=config.RF_N_ESTIMATORS):
    csv_path = str(csv_path or config.MEALS_CSV)
    print(f"\n{'='*60}\n  STUDENT MEAL PREDICTION — TRAINING PIPELINE\n{'='*60}")

    # Check if retrain is needed
    if artefact_manager.should_retrain(max_age_days=7):
        logger.info("Proceeding with training...")
    else:
        logger.info("Model is recent — training anyway (forced by direct call)")

    # 1. Load data
    print("\n[1/5] Loading data...")
    df = data_loader.load(csv_path)

    # 2. Encode
    print("\n[2/5] Encoding features...")
    df_enc, encoders = encode_features(df)
    X = df_enc[config.FEATURE_COLS]
    y = df_enc['choice']
    pos_pct = y.mean() * 100
    print(f"  Shape: {X.shape}  |  Chosen: {pos_pct:.1f}%  |  Not chosen: {100-pos_pct:.1f}%")

    # 3. Cross-validation
    print("\n[3/5] Cross-validation...")
    clf_cv = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=config.RF_MAX_DEPTH,
        min_samples_leaf=config.RF_MIN_LEAF, class_weight='balanced',
        random_state=42, n_jobs=-1,
    )
    cross_validate_model(clf_cv, X, y, n_folds)

    # 4. Train + calibrate threshold
    print("[4/5] Training final model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=config.RF_MAX_DEPTH,
        min_samples_leaf=config.RF_MIN_LEAF, class_weight='balanced',
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    print("\n  Calibrating decision threshold...")
    threshold = find_optimal_threshold(clf, X_test, y_test)
    evaluate_holdout(clf, X_test, y_test, threshold)

    # Refit on 100% before saving
    logger.info("Refitting on 100% of data before saving...")
    clf.fit(X, y)

    # 5. Save
    print("\n[5/5] Saving artefacts...")
    meta = artefact_manager.save(
        clf, encoders, config.FEATURE_COLS, threshold,
        trained_on=csv_path, n_rows=len(df),
    )
    print(f"\n  Trained at : {meta['trained_at']}")
    print(f"  Threshold  : {threshold:.2f}")
    print(f"  Rows used  : {meta['n_rows']:,}")
    print("\n  Done. Run predict.py or start api.py to serve predictions.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',   default=None)
    parser.add_argument('--folds', type=int, default=config.CV_FOLDS)
    parser.add_argument('--trees', type=int, default=config.RF_N_ESTIMATORS)
    args = parser.parse_args()
    main(csv_path=args.csv, n_folds=args.folds, n_estimators=args.trees)