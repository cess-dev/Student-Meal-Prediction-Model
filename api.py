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
from datetime import date as date_type
from typing import List, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Security
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request
import time

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
# SIMPLE IN-MEMORY CACHE
# ============================================================================
_student_id_cache: dict = {}
_context_cache: dict    = {}
CONTEXT_CACHE_TTL       = 3600   # 1 hour


def get_cached_student_id(school_id: str) -> int:
    if school_id not in _student_id_cache:
        from database import get_or_create_student
        _student_id_cache[school_id] = get_or_create_student(school_id)
    return _student_id_cache[school_id]


def get_cached_context(student_id: int, week: int):
    key = (student_id, week)
    now = time.time()
    if key in _context_cache:
        df, cached_at = _context_cache[key]
        if now - cached_at < CONTEXT_CACHE_TTL:
            return df
        del _context_cache[key]
    df = load_student_context(None, student_id, week)
    _context_cache[key] = (df, now)
    return df


# ============================================================================
# DAILY MENU HELPER
# ============================================================================
def get_available_items(date_str: str, meal_type: str) -> list:
    """
    Returns items available for a meal on a given date.
    Queries the daily_menu table for available=True items.
    Falls back to full config menu if no daily menu has been published yet.
    """
    try:
        from database import get_available_items as db_get_items
        items = db_get_items(date_str, meal_type)
        if items:
            return items
    except Exception as e:
        logger.warning(f"Daily menu lookup failed, using full menu: {e}")

    # Fallback — use full config menu
    return config.MENU_ITEMS[meal_type]


# ============================================================================
# GLOBAL STATE
# ============================================================================
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from database import create_tables
        create_tables()
        logger.info("Database tables verified ✓")
    except Exception as e:
        logger.warning(f"Database setup skipped: {e}")

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
        "**Example:** `curl -H 'X-API-Key: your-key' "
        "/predict/week?school_id=s13/34556/25&week=2`"
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
    school_id: str = Field(..., description="Student school ID e.g. s13/34556/25")
    day:       str = Field(..., description="Day: Mon, Tue, Wed, Thu, Fri, Sat, Sun")
    meal_type: str = Field(..., description="Meal type: Breakfast, Lunch, or Dinner")
    menu_item: str = Field(..., description="Exact menu item name e.g. 'Ugali Beef'")
    week:      int = Field(2, ge=1, description="Week number — use 2+ for historical features")

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
                "school_id": "s13/34556/25",
                "day": "Mon",
                "meal_type": "Lunch",
                "menu_item": "Ugali Beef",
                "week": 2,
            }
        }


class PredictionResponse(BaseModel):
    """Result of a single item prediction."""
    school_id:   str   = Field(..., description="Student school ID")
    student_id:  int   = Field(..., description="Internal student ID")
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
    predicted_orders: int   = Field(..., description="Students predicted to order this")
    pct_of_students:  float = Field(..., description="Percentage of all students")
    avg_probability:  float = Field(..., description="Average model confidence")


class HealthResponse(BaseModel):
    status:     str
    version:    str
    trained_at: str
    threshold:  float
    n_rows:     int


# ── Daily menu schemas ────────────────────────────────────────────────────────
class DailyMenuRequest(BaseModel):
    """Published by cafeteria staff each morning."""
    date:      str       = Field(..., description="Date in YYYY-MM-DD format e.g. 2026-03-21")
    day:       str       = Field(..., description="Day abbreviation: Mon, Tue, Wed, Thu, Fri, Sat, Sun")
    week:      int       = Field(..., ge=1, description="Week number")
    meal_type: str       = Field(..., description="Meal type: Breakfast, Lunch, or Dinner")
    items:     List[str] = Field(..., description="List of available menu items for this meal")

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

    @field_validator('items')
    @classmethod
    def check_items(cls, v, info):
        meal_type = info.data.get('meal_type')
        if meal_type:
            valid = config.MENU_ITEMS.get(meal_type, [])
            bad   = [i for i in v if i not in valid]
            if bad:
                raise ValueError(f"Unknown items for {meal_type}: {bad}. Valid: {valid}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "date":      "2026-03-21",
                "day":       "Mon",
                "week":      2,
                "meal_type": "Lunch",
                "items":     ["Rice Beans", "Ugali Beef", "Chapo Ndengu"],
            }
        }


class SoldOutRequest(BaseModel):
    """Marks a specific item as sold out for a meal on a given date."""
    date:      str = Field(..., description="Date in YYYY-MM-DD format e.g. 2026-03-21")
    meal_type: str = Field(..., description="Meal type: Breakfast, Lunch, or Dinner")
    menu_item: str = Field(..., description="Item that has sold out")

    class Config:
        json_schema_extra = {
            "example": {
                "date":      "2026-03-21",
                "meal_type": "Lunch",
                "menu_item": "Ugali Beef",
            }
        }


class DailyMenuRow(BaseModel):
    """One row in a daily menu response."""
    meal_type: str  = Field(..., description="Meal type")
    menu_item: str  = Field(..., description="Menu item name")
    available: bool = Field(..., description="True = available, False = sold out")


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
    response_model=MenuResponse,
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
# DAILY MENU ENDPOINTS
# ============================================================================

@app.post(
    "/menu/daily",
    tags=["Menu Management"],
    summary="Publish today's available menu",
    description=(
        "Called by cafeteria staff each morning to publish which items are "
        "available for each meal. Only published items will appear in predictions. "
        "If not called, the full config menu is used as a fallback."
    ),
)
@limiter.limit("30/minute")
def publish_daily_menu(
    request: Request,
    body: DailyMenuRequest,
    _key: str = Depends(require_api_key),
):
    try:
        from database import publish_daily_menu as db_publish
        db_publish(
            date_str=body.date,
            day=body.day,
            week=body.week,
            meal_type=body.meal_type,
            items=body.items,
        )
        return {
            "status":    "published",
            "date":      body.date,
            "meal_type": body.meal_type,
            "items":     body.items,
            "count":     len(body.items),
        }
    except Exception as e:
        logger.error(f"Failed to publish daily menu: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.patch(
    "/menu/soldout",
    tags=["Menu Management"],
    summary="Mark an item as sold out",
    description=(
        "Called by cafeteria staff in real time when an item runs out. "
        "The item is immediately excluded from all subsequent predictions for that date."
    ),
)
@limiter.limit("60/minute")
def mark_sold_out(
    request: Request,
    body: SoldOutRequest,
    _key: str = Depends(require_api_key),
):
    try:
        from database import mark_item_sold_out
        mark_item_sold_out(
            date_str=body.date,
            meal_type=body.meal_type,
            menu_item=body.menu_item,
        )
        return {
            "status":    "sold_out",
            "date":      body.date,
            "meal_type": body.meal_type,
            "menu_item": body.menu_item,
        }
    except Exception as e:
        logger.error(f"Failed to mark sold out: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/menu/daily",
    tags=["Menu Management"],
    response_model=List[DailyMenuRow],
    summary="Get today's published menu",
    description=(
        "Returns what's available and sold out for a given date and meal type. "
        "No auth required — students can use this to see what's on offer."
    ),
)
def get_daily_menu(
    date:      str = Query(..., description="Date in YYYY-MM-DD format e.g. 2026-03-21"),
    meal_type: str = Query(..., description="Meal type: Breakfast, Lunch, or Dinner"),
):
    if meal_type not in config.MEAL_TYPES:
        raise HTTPException(422, f"Invalid meal_type. Valid: {config.MEAL_TYPES}")
    try:
        from database import get_daily_menu as db_get_menu
        rows = db_get_menu(date, meal_type)
        if not rows:
            # No menu published — return full config menu as available
            return [
                {"meal_type": meal_type, "menu_item": item, "available": True}
                for item in config.MENU_ITEMS[meal_type]
            ]
        return rows
    except Exception as e:
        logger.error(f"Failed to get daily menu: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    student_id = get_cached_student_id(req.school_id)
    model, encoders, feature_cols, threshold = get_model()

    try:
        validate_inputs(encoders, req.day, req.meal_type, req.menu_item)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    ctx  = get_cached_context(student_id, req.week)
    feat = get_feature_context(ctx, req.day, req.meal_type, req.menu_item)

    result = predict_single(
        model, encoders, feature_cols,
        student_id=student_id, day=req.day,
        meal_type=req.meal_type, menu_item=req.menu_item,
        week=req.week,
        hist_freq=feat['hist_freq'],
        item_popularity=feat['item_popularity'],
        recency=feat['recency'],
        meal_streak=feat['meal_streak'],
        threshold=threshold,
    )

    prediction_logger.log_prediction(
        student_id=student_id, week=req.week, day=req.day,
        meal_type=req.meal_type, menu_item=req.menu_item,
        probability=result['probability'], choice=result['choice'],
        threshold=threshold, source='api-single',
    )
    result['school_id'] = req.school_id
    return result


@app.get(
    "/predict/menu",
    response_model=List[RankedMenuItem],
    tags=["Predictions"],
    summary="Ranked menu for one meal sitting",
    description=(
        "Returns available items for a meal sitting ranked by predicted probability. "
        "Only shows items published in the daily menu — sold out items are excluded. "
        "Falls back to the full menu if no daily menu has been published."
    ),
)
@limiter.limit("60/minute")
def predict_menu(
    request: Request,
    school_id: str = Query(..., description="Student school ID e.g. s13/34556/25"),
    day:       str = Query(..., description="Day: Mon, Tue, Wed, Thu, Fri, Sat, Sun"),
    meal_type: str = Query(..., description="Meal type: Breakfast, Lunch, or Dinner"),
    week:      int = Query(2, ge=1, description="Week number (use 2+ for best results)"),
    date:      Optional[str] = Query(None, description="Date YYYY-MM-DD to filter by daily menu"),
    _key: str = Depends(require_api_key),
):
    student_id = get_cached_student_id(school_id)
    model, encoders, feature_cols, threshold = get_model()

    if day not in config.DAYS:
        raise HTTPException(422, f"Invalid day '{day}'. Valid: {config.DAYS}")
    if meal_type not in config.MEAL_TYPES:
        raise HTTPException(422, f"Invalid meal_type '{meal_type}'. Valid: {config.MEAL_TYPES}")

    # Get only available items for this date
    available = get_available_items(date or str(date_type.today()), meal_type)

    df = predict_meal_options(
        model, encoders, feature_cols,
        student_id, day, meal_type, week, threshold,
        csv_path=None,
        available_items=available,
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
    school_id: str = Query(..., description="Student school ID e.g. s13/34556/25"),
    week:      int = Query(2, ge=1, description="Week number (use 2+ for best results)"),
    _key: str = Depends(require_api_key),
):
    student_id = get_cached_student_id(school_id)
    model, encoders, feature_cols, threshold = get_model()

    df = predict_weekly(
        model, encoders, feature_cols,
        student_id, week, threshold,
        csv_path=None,
    )
    return df.to_dict(orient='records')


@app.get(
    "/predict/batch",
    response_model=List[BatchSummaryRow],
    tags=["Predictions"],
    summary="Cafeteria portion estimate",
    description=(
        "For a given day and meal type, estimates how many students will order each item. "
        "Only includes items available in the daily menu for that date. "
        "Use this for cafeteria planning and stock ordering."
    ),
)
@limiter.limit("5/minute")
def predict_batch(
    request: Request,
    day:       str = Query(..., description="Day: Mon, Tue, Wed, Thu, Fri, Sat, Sun"),
    meal_type: str = Query(..., description="Meal type: Breakfast, Lunch, or Dinner"),
    week:      int = Query(2, ge=1, description="Week number"),
    date:      Optional[str] = Query(None, description="Date YYYY-MM-DD to filter by daily menu"),
    _key: str = Depends(require_api_key),
):
    model, encoders, feature_cols, threshold = get_model()

    if day not in config.DAYS:
        raise HTTPException(422, f"Invalid day '{day}'. Valid: {config.DAYS}")
    if meal_type not in config.MEAL_TYPES:
        raise HTTPException(422, f"Invalid meal_type '{meal_type}'. Valid: {config.MEAL_TYPES}")

    # Only score items available today — sold out items are excluded
    items        = get_available_items(date or str(date_type.today()), meal_type)
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

    df_input                 = pd.DataFrame(records)
    X                        = df_input[feature_cols]
    probas                   = model.predict_proba(X)[:, 1]
    df_input['probability']  = probas
    df_input['choice']       = (probas >= threshold).astype(int)

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