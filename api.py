"""
api.py
======
FastAPI serving layer. Loads model once at startup.

Run:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
Docs:
    http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from typing import List

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Security
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request

import artefact_manager
import config
import prediction_logger
from predict import (
    predict_meal_options,
    predict_single,
    predict_weekly,
    validate_inputs,
    load_student_context,
    get_feature_context,
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
    description=(
        "Predicts which cafeteria meals a student is likely to order "
        "based on their historical ordering patterns.\n\n"
        "**Authentication:** All prediction endpoints require an `X-API-Key` header.\n\n"
        "**Example:** `curl -H 'X-API-Key: your-key' /predict/week?student_id=1&week=2`"
    ),
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
# REQUEST / RESPONSE SCHEMAS
# ============================================================================
class SinglePredictionRequest(BaseModel):
    """Request body for a single item prediction."""
    student_id: int = Field(..., gt=0, description="Student ID (positive integer)")
    day:        str = Field(..., description="Day abbreviation: Mon, Tue, Wed, Thu, Fri, Sat, Sun")
    meal_type:  str = Field(..., description="Meal type: Breakfast, Lunch, or Dinner")
    menu_item:  str = Field(..., description="Exact menu item name e.g. 'Ugali Beef'")
    week:       int = Field(2, ge=1, description="Week number — use 2+ for historical features")

    @field_validator('day')
    @classmethod
    def check_day(cls, v):
        if v not in config.DAYS:
            raise ValueError(f"Invalid day '{v}'. Valid options: {config.DAYS}")
        return v

    @field_validator('meal_type')
    @classmethod
    def check_meal_type(cls, v):
        if v not in config.MEAL_TYPES:
            raise ValueError(f"Invalid meal_type '{v}'. Valid options: {config.MEAL_TYPES}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "student_id": 1,
                "day": "Mon",
                "meal_type": "Lunch",
                "menu_item": "Ugali Beef",
                "week": 2,
            }
        }


class PredictionResponse(BaseModel):
    """Result of a single item prediction."""
    student_id:  int   = Field(..., description="Student ID")
    day:         str   = Field(..., description="Day of week")
    meal_type:   str   = Field(..., description="Meal type")
    menu_item:   str   = Field(..., description="Menu item predicted")
    choice:      int   = Field(..., description="1 = will order, 0 = won't order")
    probability: float = Field(..., description="Model confidence (0.0 to 1.0)")
    label:       str   = Field(..., description="Human readable prediction label")

class MenuResponse(BaseModel):
    Breakfast: List[str] = Field(..., description="Breakfast menu items")
    Lunch:     List[str] = Field(..., description="Lunch menu items")
    Dinner:    List[str] = Field(..., description="Dinner menu items")

class RankedMenuItem(BaseModel):
    """One item in a ranked menu response."""
    rank:        int   = Field(..., description="Rank by probability (1 = most likely)")
    menu_item:   str   = Field(..., description="Menu item name")
    probability: float = Field(..., description="Predicted probability (0.0 to 1.0)")
    label:       str   = Field(..., description="Prediction label")


class WeeklyForecastRow(BaseModel):
    """One row in a weekly forecast."""
    day:              str   = Field(..., description="Day of week")
    meal_type:        str   = Field(..., description="Meal type")
    menu_item:        str   = Field(..., description="Predicted menu item")
    probability:      float = Field(..., description="Model confidence")
    predicted_choice: str   = Field(..., description="Will order / Best guess label")
    attendance:       str   = Field(..., description="Expected attendance likelihood")


class BatchSummaryRow(BaseModel):
    """One row in a batch/cafeteria summary."""
    menu_item:        str   = Field(..., description="Menu item name")
    predicted_orders: int   = Field(..., description="Number of students predicted to order this")
    pct_of_students:  float = Field(..., description="Percentage of all students")
    avg_probability:  float = Field(..., description="Average model confidence across all students")


class HealthResponse(BaseModel):
    status:     str
    version:    str
    trained_at: str
    threshold:  float
    n_rows:     int


# ============================================================================
# SYSTEM ENDPOINTS
# ============================================================================

@app.get(
    "/health",
    tags=["System"],
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the current status of the API and model. No auth required.",
)
def health():
    if 'error' in _state:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "detail": _state['error']}
        )
    meta = _state.get('meta', {})
    return {
        "status":     "healthy",
        "version":    app.version,
        "trained_at": meta.get('trained_at', 'unknown'),
        "threshold":  _state.get('threshold', 0),
        "n_rows":     meta.get('n_rows', 0),
    }


@app.get(
    "/menu",
    tags=["Reference"],
    response_model= MenuResponse,
    summary="Get full menu",
    description="Returns all valid menu items grouped by meal type. No auth required.",
)
def get_menu():
    return config.MENU_ITEMS


@app.get(
    "/days",
    tags=["Reference"],
    response_model=List[str],
    summary="Get valid days",
    description="Returns valid day abbreviations in week order. No auth required.",
)
def get_days():
    return config.DAYS


# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Single item prediction",
    description=(
        "Predict whether a specific student will order a specific menu item "
        "on a given day and meal. The API automatically looks up the student's "
        "historical features from the database."
    ),
)
@limiter.limit("60/minute")
def predict(
    request: Request,
    req: SinglePredictionRequest,
    _key: str = Depends(require_api_key),
):
    model, encoders, feature_cols, threshold = get_model()

    try:
        validate_inputs(encoders, req.day, req.meal_type, req.menu_item)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Look up real feature context from database — caller does not supply these
    ctx  = load_student_context(None, req.student_id, req.week)
    feat = get_feature_context(ctx, req.day, req.meal_type, req.menu_item)

    result = predict_single(
        model, encoders, feature_cols,
        student_id=req.student_id, day=req.day,
        meal_type=req.meal_type, menu_item=req.menu_item,
        week=req.week,
        hist_freq=feat['hist_freq'],
        item_popularity=feat['item_popularity'],
        recency=feat['recency'],
        meal_streak=feat['meal_streak'],
        threshold=threshold,
    )

    prediction_logger.log_prediction(
        student_id=req.student_id, week=req.week, day=req.day,
        meal_type=req.meal_type, menu_item=req.menu_item,
        probability=result['probability'], choice=result['choice'],
        threshold=threshold, source='api-single',
    )
    return result


@app.get(
    "/predict/menu",
    response_model=List[RankedMenuItem],
    tags=["Predictions"],
    summary="Ranked menu for one meal sitting",
    description=(
        "Returns all items for a given meal sitting ranked by predicted probability. "
        "Use this to show a personalised menu to a student before they order."
    ),
)
@limiter.limit("60/minute")
def predict_menu(
    request: Request,
    student_id: int = Query(..., gt=0, description="Student ID"),
    day:        str = Query(..., description="Day: Mon, Tue, Wed, Thu, Fri, Sat, Sun"),
    meal_type:  str = Query(..., description="Meal type: Breakfast, Lunch, or Dinner"),
    week:       int = Query(2,   ge=1, description="Week number (use 2+ for best results)"),
    _key: str = Depends(require_api_key),
):
    model, encoders, feature_cols, threshold = get_model()

    if day not in config.DAYS:
        raise HTTPException(422, f"Invalid day '{day}'. Valid: {config.DAYS}")
    if meal_type not in config.MEAL_TYPES:
        raise HTTPException(422, f"Invalid meal_type '{meal_type}'. Valid: {config.MEAL_TYPES}")

    df = predict_meal_options(
        model, encoders, feature_cols,
        student_id, day, meal_type, week, threshold,
        csv_path=None,   # always use DB, never accept path from caller
    )
    return df.to_dict(orient='records')


@app.get(
    "/predict/week",
    response_model=List[WeeklyForecastRow],
    tags=["Predictions"],
    summary="Full weekly forecast for a student",
    description=(
        "Returns the most likely meal choice per (day × meal type) for an entire week. "
        "Includes attendance confidence and applies a diversity constraint so the same "
        "item is not predicted every day."
    ),
)
@limiter.limit("20/minute")
def predict_week(
    request: Request,
    student_id: int = Query(..., gt=0, description="Student ID"),
    week:       int = Query(2,   ge=1, description="Week number (use 2+ for best results)"),
    _key: str = Depends(require_api_key),
):
    model, encoders, feature_cols, threshold = get_model()

    df = predict_weekly(
        model, encoders, feature_cols,
        student_id, week, threshold,
        csv_path=None,   # always use DB, never accept path from caller
    )
    return df.to_dict(orient='records')


@app.get(
    "/predict/batch",
    response_model=List[BatchSummaryRow],
    tags=["Predictions"],
    summary="Cafeteria portion estimate",
    description=(
        "For a given day and meal type, estimates how many students will order each item. "
        "Use this for cafeteria planning and stock ordering. "
        "Scores all students in a single model call for efficiency."
    ),
)
@limiter.limit("5/minute")
def predict_batch(
    request: Request,
    day:       str = Query(..., description="Day: Mon, Tue, Wed, Thu, Fri, Sat, Sun"),
    meal_type: str = Query(..., description="Meal type: Breakfast, Lunch, or Dinner"),
    week:      int = Query(2,   ge=1, description="Week number"),
    _key: str = Depends(require_api_key),
):
    model, encoders, feature_cols, threshold = get_model()

    if day not in config.DAYS:
        raise HTTPException(422, f"Invalid day '{day}'. Valid: {config.DAYS}")
    if meal_type not in config.MEAL_TYPES:
        raise HTTPException(422, f"Invalid meal_type '{meal_type}'. Valid: {config.MEAL_TYPES}")

    items        = config.MENU_ITEMS[meal_type]
    num_students = config.NUM_STUDENTS
    day_enc      = encoders['day'].transform([day])[0]
    mt_enc       = encoders['meal_type'].transform([meal_type])[0]
    day_num      = config.DAY_ORDER[day]

    records = []
    for sid in range(1, num_students + 1):
        for item in items:
            item_enc = encoders['menu_item'].transform([item])[0]
            cs       = artefact_manager.get_coldstart_features(meal_type, item)
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

    df_input             = pd.DataFrame(records)
    X                    = df_input[feature_cols]
    probas               = model.predict_proba(X)[:, 1]
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