"""
tests/test_pipeline.py
"""

import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import data_loader
import artefact_manager
from predict import (
    apply_diversity,
    build_inference_row,
    get_feature_context,
    load_student_context,
    predict_meal_options,
    predict_single,
    predict_weekly,
    validate_inputs,
)


@pytest.fixture(scope='session')
def small_df():
    return data_loader.generate_synthetic_data(num_students=10, num_weeks=2, seed=99)

@pytest.fixture(scope='session')
def engineered_df(small_df):
    valid = data_loader.validate(small_df)
    return data_loader.engineer_features(valid)

@pytest.fixture(scope='session')
def trained_artefacts(engineered_df, tmp_path_factory):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    df = engineered_df.copy()
    encoders = {}
    for col in config.CAT_COLS:
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col])
        encoders[col] = le

    X = df[config.FEATURE_COLS]
    y = df['choice']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=20, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    probas    = clf.predict_proba(X_test)[:, 1]
    threshold = float(np.percentile(probas, 85))
    return clf, encoders, config.FEATURE_COLS, threshold


class TestDataLoader:
    def test_shape(self, small_df):
        assert len(small_df) > 1000

    def test_columns(self, small_df):
        assert set(small_df.columns) == {'week','student_id','day','meal_type','menu_item','choice'}

    def test_choice_binary(self, small_df):
        assert set(small_df['choice'].unique()).issubset({0, 1})

    def test_one_choice_per_meal_sitting(self, small_df):
        chosen = small_df[small_df['choice'] == 1]
        counts = chosen.groupby(['week','student_id','day','meal_type']).size()
        assert (counts <= 1).all()

    def test_valid_days(self, small_df):
        assert set(small_df['day'].unique()).issubset(set(config.DAYS))

    def test_valid_meal_types(self, small_df):
        assert set(small_df['meal_type'].unique()).issubset(set(config.MEAL_TYPES))

    def test_validation_rejects_bad_day(self, small_df):
        bad = small_df.copy()
        bad.loc[0, 'day'] = 'Funday'
        with pytest.raises(ValueError, match="day"):
            data_loader.validate(bad)

    def test_validation_rejects_nulls(self, small_df):
        bad = small_df.copy()
        bad.loc[0, 'student_id'] = None
        with pytest.raises(ValueError):
            data_loader.validate(bad)

    def test_feature_engineering_columns(self, engineered_df):
        expected = {'hist_freq','item_popularity','day_num','recency','meal_streak'}
        assert expected.issubset(set(engineered_df.columns))

    def test_hist_freq_no_leakage(self, engineered_df):
        """
        The true no-leakage invariant: the very first time a student x item
        combination appears, hist_freq must be 0. It is correct for hist_freq
        to be > 0 later in week 1 if the item appeared earlier that week.
        """
        first_occurrences = (
            engineered_df
            .sort_values(['week', 'student_id', 'day_num', 'meal_type'])
            .groupby(['student_id', 'menu_item'])
            .first()
            .reset_index()
        )
        assert (first_occurrences['hist_freq'] == 0).all(), \
            "hist_freq is non-zero on first occurrence — data leakage detected"

    def test_item_popularity_range(self, engineered_df):
        assert engineered_df['item_popularity'].between(0, 1).all()


class TestPredict:
    def test_validate_inputs_passes(self, trained_artefacts):
        _, encoders, _, _ = trained_artefacts
        validate_inputs(encoders, 'Mon', 'Breakfast', 'Eggs')

    def test_validate_inputs_bad_day(self, trained_artefacts):
        _, encoders, _, _ = trained_artefacts
        with pytest.raises(ValueError, match="day"):
            validate_inputs(encoders, 'Monday', 'Breakfast', 'Eggs')

    def test_validate_inputs_bad_item(self, trained_artefacts):
        _, encoders, _, _ = trained_artefacts
        with pytest.raises(ValueError, match="menu_item"):
            validate_inputs(encoders, 'Mon', 'Breakfast', 'Pizza')

    def test_predict_single_returns_dict(self, trained_artefacts):
        model, encoders, feature_cols, threshold = trained_artefacts
        result = predict_single(model, encoders, feature_cols,
                                student_id=1, day='Mon', meal_type='Breakfast',
                                menu_item='Eggs', threshold=threshold)
        assert isinstance(result, dict)
        assert 'probability' in result
        assert 'choice' in result
        assert result['choice'] in (0, 1)
        assert 0.0 <= result['probability'] <= 1.0

    def test_predict_single_probability_range(self, trained_artefacts):
        model, encoders, feature_cols, threshold = trained_artefacts
        for item in config.MENU_ITEMS['Lunch']:
            result = predict_single(model, encoders, feature_cols,
                                    student_id=3, day='Wed', meal_type='Lunch',
                                    menu_item=item, threshold=threshold)
            assert 0.0 <= result['probability'] <= 1.0

    def test_predict_meal_options_length(self, trained_artefacts):
        model, encoders, feature_cols, threshold = trained_artefacts
        df = predict_meal_options(model, encoders, feature_cols,
                                  student_id=1, day='Mon', meal_type='Lunch',
                                  threshold=threshold)
        assert len(df) == len(config.MENU_ITEMS['Lunch'])

    def test_predict_meal_options_sorted(self, trained_artefacts):
        model, encoders, feature_cols, threshold = trained_artefacts
        df = predict_meal_options(model, encoders, feature_cols,
                                  student_id=1, day='Mon', meal_type='Lunch',
                                  threshold=threshold)
        probs = df['probability'].tolist()
        assert probs == sorted(probs, reverse=True)

    def test_predict_weekly_shape(self, trained_artefacts):
        model, encoders, feature_cols, threshold = trained_artefacts
        df = predict_weekly(model, encoders, feature_cols,
                            student_id=1, week=2, threshold=threshold)
        assert len(df) == 21

    def test_predict_weekly_diversity(self, trained_artefacts):
        model, encoders, feature_cols, threshold = trained_artefacts
        df = predict_weekly(model, encoders, feature_cols,
                            student_id=1, week=2, threshold=threshold)
        for meal_type in config.MEAL_TYPES:
            items = df[df['meal_type'] == meal_type]['menu_item'].tolist()
            assert len(set(items)) > 1, (
                f"All 7 days predicted same item for {meal_type} — diversity not working")

    def test_diversity_penalty_applied(self):
        items = [
            {'menu_item': 'Ugali Beef',   'probability': 0.30, 'choice': 0, 'label': '→'},
            {'menu_item': 'Rice Beans',   'probability': 0.25, 'choice': 0, 'label': '→'},
            {'menu_item': 'Chapo Ndengu', 'probability': 0.20, 'choice': 0, 'label': '→'},
        ]
        result = apply_diversity(items, {'Ugali Beef'}, penalty=0.5)
        ugali  = next(x for x in result if x['menu_item'] == 'Ugali Beef')
        assert ugali['probability'] == pytest.approx(0.15, abs=0.01)

    def test_coldstart_fallback(self):
        feats = artefact_manager.get_coldstart_features('Lunch', 'Ugali Beef')
        assert feats['hist_freq'] == 0.0
        assert 0.0 < feats['item_popularity'] < 1.0


class TestArtefactManager:
    def test_save_and_load(self, trained_artefacts, tmp_path, monkeypatch):
        model, encoders, feature_cols, threshold = trained_artefacts
        monkeypatch.setattr(config, 'MODEL_DIR',     tmp_path)
        monkeypatch.setattr(config, 'MODEL_PATH',    tmp_path / 'meal_model.pkl')
        monkeypatch.setattr(config, 'ENCODERS_PATH', tmp_path / 'encoders.pkl')
        monkeypatch.setattr(config, 'FEATURES_PATH', tmp_path / 'feature_cols.pkl')
        monkeypatch.setattr(config, 'THRESHOLD_PATH',tmp_path / 'threshold.pkl')
        monkeypatch.setattr(config, 'METADATA_PATH', tmp_path / 'model_metadata.json')
        artefact_manager.save(model, encoders, feature_cols, threshold,
                              trained_on='test.csv', n_rows=500)
        loaded_model, loaded_enc, loaded_fc, loaded_t, meta = artefact_manager.load()
        assert loaded_t == pytest.approx(threshold, abs=0.001)
        assert loaded_fc == feature_cols
        assert meta['n_rows'] == 500

    def test_integrity_check_catches_tamper(self, trained_artefacts, tmp_path, monkeypatch):
        model, encoders, feature_cols, threshold = trained_artefacts
        monkeypatch.setattr(config, 'MODEL_DIR',     tmp_path)
        monkeypatch.setattr(config, 'MODEL_PATH',    tmp_path / 'meal_model.pkl')
        monkeypatch.setattr(config, 'ENCODERS_PATH', tmp_path / 'encoders.pkl')
        monkeypatch.setattr(config, 'FEATURES_PATH', tmp_path / 'feature_cols.pkl')
        monkeypatch.setattr(config, 'THRESHOLD_PATH',tmp_path / 'threshold.pkl')
        monkeypatch.setattr(config, 'METADATA_PATH', tmp_path / 'model_metadata.json')
        artefact_manager.save(model, encoders, feature_cols, threshold)
        with open(tmp_path / 'encoders.pkl', 'wb') as f:
            pickle.dump({'tampered': True}, f)
        with pytest.raises(RuntimeError, match="integrity check FAILED"):
            artefact_manager.load()

    def test_should_retrain_missing_metadata(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, 'METADATA_PATH', tmp_path / 'nonexistent.json')
        assert artefact_manager.should_retrain() is True
