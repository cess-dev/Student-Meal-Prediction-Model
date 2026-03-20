"""
api.py
======
FastAPI serving layer. Loads model once at startup.

Run:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
Docs:
    http://localhost:8000/docs
"""

import time
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Security
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request

import artefact_manager
import config
import data_loader
import prediction_logger
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

logger  = config.get_logger(__name__)
limiter = Limiter(key_func=get_remote_address)

# ============================================================================
# GLOBAL STATE
# ============================================================================
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model, encoders, feature_cols, threshold, meta = artefact_manager.load()
        _state.update(model=model, encoders=encoders,
                      feature_cols=feature_cols, threshold=threshold, meta=meta)
        logger.info(f"Model loaded ✓  trained={meta['trained_at'][:19]}  threshold={threshold:.2f}")
    except (FileNotFoundError, RuntimeError) as exc:
        _state['error'] = str(exc)
        logger.error(f"Model load failed: {exc}")
    yield
    _state.clear()


# ============================================================================
# APP
# ============================================================================
app = FastAPI(
    title="Student Meal Prediction API",
    description="Predicts cafeteria meal choices per student based on historical ordering patterns.",
    version="2.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ============================================================================
# AUTH
# ============================================================================
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def require_api_key(key: str = Security(API_KEY_HEADER)):
    if key != config.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")
    return key


# ============================================================================
# HELPERS
# ============================================================================
def get_model():
    if 'error' in _state:
        raise HTTPException(status_code=503, detail=_state['error'])
    if 'model' not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return _state['model'], _state['encoders'], _state['feature_cols'], _state['threshold']


# ============================================================================
# SCHEMAS
# ============================================================================
class SinglePredictionRequest(BaseModel):
    student_id:      int   = Field(..., gt=0)
    day:             str
    meal_type:       str
    menu_item:       str
    week:            int   = Field(2, ge=1)
    hist_freq:       float = Field(0.0, ge=0)
    item_popularity: float = Field(0.0, ge=0, le=1)
    recency:         int   = Field(0, ge=0, le=1)
    meal_streak:     int   = Field(0, ge=0)

    @field_validator('day')
    @classmethod
    def check_day(cls, v):
        if v not in config.DAYS:
            raise ValueError(f"Invalid day '{v}'. Valid: {config.DAYS}")
        return v

    @field_validator('meal_type')
    @classmethod
    def check_meal_type(cls, v):
        if v not in config.MEAL_TYPES:
            raise ValueError(f"Invalid meal_type '{v}'. Valid: {config.MEAL_TYPES}")
        return v


class PredictionResponse(BaseModel):
    student_id: int; day: str; meal_type: str; menu_item: str
    choice: int; probability: float; label: str


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health", tags=["System"])
def health():
    if 'error' in _state:
        return JSONResponse(status_code=503, content={"status":"unhealthy","detail":_state['error']})
    meta = _state.get('meta', {})
    return {
        "status":     "healthy",
        "version":    app.version,
        "trained_at": meta.get('trained_at', 'unknown'),
        "threshold":  _state.get('threshold'),
        "n_rows":     meta.get('n_rows'),
    }


@app.get("/menu", tags=["Reference"])
def get_menu():
    return config.MENU_ITEMS


@app.get("/days", tags=["Reference"])
def get_days():
    return config.DAYS


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
@limiter.limit("60/minute")
def predict(request: Request, req: SinglePredictionRequest,
            _key: str = Depends(require_api_key)):
    model, encoders, feature_cols, threshold = get_model()
    try:
        validate_inputs(encoders, req.day, req.meal_type, req.menu_item)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    result = predict_single(
        model, encoders, feature_cols,
        student_id=req.student_id, day=req.day, meal_type=req.meal_type,
        menu_item=req.menu_item, week=req.week,
        hist_freq=req.hist_freq, item_popularity=req.item_popularity,
        recency=req.recency, meal_streak=req.meal_streak, threshold=threshold,
    )
    prediction_logger.log_prediction(
        student_id=req.student_id, week=req.week, day=req.day,
        meal_type=req.meal_type, menu_item=req.menu_item,
        probability=result['probability'], choice=result['choice'],
        threshold=threshold, source='api-single',
    )
    return result


@app.get("/predict/menu", tags=["Predictions"])
@limiter.limit("60/minute")
def predict_menu(
    request: Request,
    student_id: int = Query(..., gt=0),
    day:        str = Query(...),
    meal_type:  str = Query(...),
    week:       int = Query(2, ge=1),
    csv:        str = Query(None),
    _key: str = Depends(require_api_key),
):
    model, encoders, feature_cols, threshold = get_model()
    if day not in config.DAYS:
        raise HTTPException(422, f"Invalid day '{day}'. Valid: {config.DAYS}")
    if meal_type not in config.MEAL_TYPES:
        raise HTTPException(422, f"Invalid meal_type. Valid: {config.MEAL_TYPES}")
    df = predict_meal_options(model, encoders, feature_cols,
                               student_id, day, meal_type, week, threshold, csv)
    return df.to_dict(orient='records')


@app.get("/predict/week", tags=["Predictions"])
@limiter.limit("20/minute")
def predict_week(
    request: Request,
    student_id: int = Query(..., gt=0),
    week:       int = Query(2, ge=1),
    csv:        str = Query(None),
    _key: str = Depends(require_api_key),
):
    model, encoders, feature_cols, threshold = get_model()
    df = predict_weekly(model, encoders, feature_cols,
                        student_id, week, threshold, csv)
    return df.to_dict(orient='records')


@app.get("/predict/batch", tags=["Predictions"])
@limiter.limit("5/minute")
def predict_batch(
    request: Request,
    day:       str = Query(...),
    meal_type: str = Query(...),
    week:      int = Query(2, ge=1),
    _key: str = Depends(require_api_key),
):
    """
    Vectorised batch prediction — estimates how many students will order each
    item. Scores all students in a single predict_proba() call (not a loop).
    """
    model, encoders, feature_cols, threshold = get_model()
    if day not in config.DAYS:
        raise HTTPException(422, f"Invalid day '{day}'.")
    if meal_type not in config.MEAL_TYPES:
        raise HTTPException(422, f"Invalid meal_type.")

    items        = config.MENU_ITEMS[meal_type]
    num_students = config.NUM_STUDENTS
    day_enc      = encoders['day'].transform([day])[0]
    mt_enc       = encoders['meal_type'].transform([meal_type])[0]
    day_num      = config.DAY_ORDER[day]

    # Build one large matrix: num_students × num_items rows, scored in one shot
    records = []
    for sid in range(1, num_students + 1):
        for item in items:
            item_enc = encoders['menu_item'].transform([item])[0]
            cs = artefact_manager.get_coldstart_features(meal_type, item)
            records.append({
                'student_id':      sid,
                'week':            week,
                'day_enc':         day_enc,
                'meal_type_enc':   mt_enc,
                'menu_item_enc':   item_enc,
                'hist_freq':       cs['hist_freq'],
                'item_popularity': cs['item_popularity'],
                'recency':         cs['recency'],
                'meal_streak':     cs['meal_streak'],
                'day_num':         day_num,
                '_item':           item,
            })

    df_input = pd.DataFrame(records)
    X        = df_input[feature_cols]
    probas   = model.predict_proba(X)[:, 1]
    df_input['probability'] = probas
    df_input['choice']      = (probas >= threshold).astype(int)

    summary = []
    for item in items:
        mask  = df_input['_item'] == item
        count = int(df_input.loc[mask, 'choice'].sum())
        summary.append({
            'menu_item':        item,
            'predicted_orders': count,
            'pct_of_students':  round(count / num_students * 100, 1),
            'avg_probability':  round(float(df_input.loc[mask, 'probability'].mean()), 4),
        })

    summary.sort(key=lambda x: x['predicted_orders'], reverse=True)
    return summary