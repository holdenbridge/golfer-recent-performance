import json
import os
import pickle
import sys
from glob import glob
from pathlib import Path

import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

GOLF_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = GOLF_DIR / "SavedData"

sys.path.insert(0, str(GOLF_DIR))
from golf_helpers import (
    load_event_odds,
    american_odds_to_implied_prob,
    estimate_finishing_position,
    calc_slope,
    recent_streak,
)

# ---------------------------------------------------------------------------
# Data loading (once at import time)
# ---------------------------------------------------------------------------

with open(DATA_DIR / "owgr_dict.json") as f:
    owgr_dict: dict = json.load(f)

with open(DATA_DIR / "tournament_fields.json") as f:
    tournament_fields: dict = json.load(f)

# Selects the most recent model by sorted date
# model_files = sorted(glob(str(GOLF_DIR / "beat_expectation_model_*.pkl")))
# if not model_files:
#     raise FileNotFoundError("No beat_expectation_model_*.pkl found in Golf Trends/")
# with open(model_files[-1], "rb") as f:
#     xgb_model = pickle.load(f)

with open(GOLF_DIR / "beat_expectation_model_03_02_26.pkl", "rb") as f:
    xgb_model = pickle.load(f)

NEXT_EVENTS: dict[str, str] = {
    "WMOpen2026": "2026-02-07",
    "ATTPebble2026": "2026-02-14",
    "Genesis2026": "2026-02-21",
    "Cognizant2026": "2026-02-28",
    "ArnoldPalmer2026": "2026-03-07",
    "PlayersChampionship2026": "2026-03-14",
}

MAJOR_DATES = {
    "PlayersChampionship2023": "2023-03-04",
    "Masters2023": "2023-04-01",
    "PGAChampionship2023": "2023-05-13",
    "USOpen2023": "2023-06-10",
    "OpenChampionship2023": "2023-07-15",
    "PlayersChampionship2024": "2024-03-09",
    "Masters2024": "2024-04-06",
    "PGAChampionship2024": "2024-05-11",
    "USOpen2024": "2024-06-08",
    "OpenChampionship2024": "2024-07-13",
    "PlayersChampionship2025": "2025-03-08",
    "Masters2025": "2025-04-05",
    "PGAChampionship2025": "2025-05-10",
    "USOpen2025": "2025-06-07",
    "OpenChampionship2025": "2025-07-12",
}

MODEL_FEATURES = [
    "Avg_Points_StartEvent",
    "Week4Change",
    "recent_vs_distant",
    "recent_positive_streak",
    "acceleration_4v12",
    "points_x_momentum8",
]

# ---------------------------------------------------------------------------
# Prediction helpers (ported from live_predict.ipynb)
# ---------------------------------------------------------------------------

def _closest_date_on_or_before(player_dates: dict, target: str) -> str | None:
    """Return the closest date key <= *target*, or None if none exists."""
    candidates = [d for d in player_dates if d <= target]
    return max(candidates) if candidates else None


def _evaluate_live_event(event_start_date: str, event_name: str) -> pd.DataFrame:
    weeks_before = range(1, 13)
    base_date = datetime.strptime(event_start_date, "%Y-%m-%d")
    pre_event_date = (base_date - timedelta(weeks=1)).strftime("%Y-%m-%d")
    week_dates = {
        f"Avg_Points_{w}weekbefore": (base_date - timedelta(weeks=w)).strftime("%Y-%m-%d")
        for w in weeks_before
    }

    player_odds = load_event_odds(event_name, str(DATA_DIR))
    rows = []
    for player, data in owgr_dict.items():
        if player not in tournament_fields.get(event_name, {}):
            continue
        row = {"PlayerName": player}
        for colname, week_date in week_dates.items():
            actual = _closest_date_on_or_before(data, week_date)
            row[colname] = data[actual].get("Avg_Points") if actual else None
        start_key = _closest_date_on_or_before(data, pre_event_date)
        row["Avg_Points_StartEvent"] = data[start_key].get("Avg_Points") if start_key else None
        if row["Avg_Points_StartEvent"] is None:
            continue
        row["EventName"] = event_name
        row["FinishingPosition"] = tournament_fields[event_name][player]["Finishing Position"]
        odds = player_odds.get(player, {})
        for col in ("WinOdds", "T5Odds", "T10Odds", "T20Odds"):
            row[col] = odds.get(col)
        rows.append(row)
    return pd.DataFrame(rows)


def _prepare_live_event(df: pd.DataFrame) -> pd.DataFrame:
    df_result = df.copy()
    for i in range(1, 13):
        week_col = f"Avg_Points_{i}weekbefore"
        df_result[f"Week{i}Change"] = df_result["Avg_Points_StartEvent"] - df_result[week_col]
    keep = [
        c for c in df_result.columns
        if "Change" in c or "Odds" in c
        or c in ("PlayerName", "Avg_Points_StartEvent", "EventName", "FinishingPosition")
    ]
    return df_result[keep]


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = (
        ["Avg_Points_StartEvent", "T20Odds"]
        + [f"Week{i}Change" for i in range(1, 13)]
    )
    df = df.dropna(subset=required_cols).copy()
    df["T20Prob"] = df["T20Odds"].apply(american_odds_to_implied_prob)
    df["EstimatedFinishingPosition"] = df.apply(estimate_finishing_position, axis=1)
    df["momentum_slope"] = df.apply(calc_slope, axis=1)
    df["recent_momentum"] = df[
        ["Week1Change", "Week2Change", "Week3Change", "Week4Change"]
    ].mean(axis=1)
    df["distant_momentum"] = df[
        ["Week9Change", "Week10Change", "Week11Change", "Week12Change"]
    ].mean(axis=1)
    df["recent_vs_distant"] = df["recent_momentum"] - df["distant_momentum"]
    df["momentum_volatility"] = df[
        [f"Week{i}Change" for i in range(1, 13)]
    ].std(axis=1)
    df["best_week"] = df[[f"Week{i}Change" for i in range(1, 13)]].max(axis=1)
    df["worst_week"] = df[[f"Week{i}Change" for i in range(1, 13)]].min(axis=1)
    df["week_range"] = df["best_week"] - df["worst_week"]
    df["positive_week_count"] = (
        df[[f"Week{i}Change" for i in range(1, 13)]] > 0
    ).sum(axis=1)
    df["recent_positive_streak"] = df.apply(recent_streak, axis=1)
    df["acceleration_4v12"] = df["Week4Change"] - df["Week12Change"]
    df["points_x_momentum8"] = df["Avg_Points_StartEvent"] * df["Week8Change"]
    df["ModelPrediction"] = xgb_model.predict_proba(df[MODEL_FEATURES])[:, 1]
    df["T20_ImpliedProb"] = df["T20Odds"].apply(american_odds_to_implied_prob)
    return df


def get_event_predictions(event_name: str) -> list[dict]:
    event_date = NEXT_EVENTS.get(event_name)
    if not event_date:
        return []
    raw = _evaluate_live_event(event_date, event_name)
    if raw.empty:
        return []
    prepared = _prepare_live_event(raw)
    df = _engineer_features(prepared)
    cols = [
        "PlayerName", "ModelPrediction",
        "WinOdds", "T5Odds", "T10Odds", "T20Odds", "T20_ImpliedProb",
        "FinishingPosition", "EstimatedFinishingPosition",
    ]
    result = df[cols].sort_values("ModelPrediction", ascending=False)
    return result.to_dict(orient="records")

# ---------------------------------------------------------------------------
# Player time-series helper (ported from golf_trends_trading.ipynb)
# ---------------------------------------------------------------------------

def _get_current_ranking(player_name: str) -> int | None:
    """Rank all players by their latest Avg_Points (descending) and return 1-based rank."""
    latest_points = []
    for name, date_map in owgr_dict.items():
        latest_date = max(date_map.keys())
        pts = date_map[latest_date].get("Avg_Points")
        if pts is not None:
            latest_points.append((name, pts))
    latest_points.sort(key=lambda x: x[1], reverse=True)
    for i, (name, _) in enumerate(latest_points, 1):
        if name == player_name:
            return i
    return None


def get_player_timeseries(player_name: str) -> dict:
    if player_name not in owgr_dict:
        return {"error": f"Player '{player_name}' not found"}
    date_points = sorted(
        ((d, v.get("Avg_Points")) for d, v in owgr_dict[player_name].items()),
        key=lambda x: x[0],
    )
    dates = [d for d, _ in date_points]
    avg_points = [v for _, v in date_points]
    is_offseason = [
        datetime.strptime(d, "%Y-%m-%d").month in (9, 10, 11, 12) for d in dates
    ]
    majors = []
    for name, mdate in MAJOR_DATES.items():
        if mdate in dates:
            idx = dates.index(mdate)
            majors.append({"date": mdate, "points": avg_points[idx], "name": name})
    return {
        "player": player_name,
        "ranking": _get_current_ranking(player_name),
        "dates": dates,
        "avg_points": avg_points,
        "is_offseason": is_offseason,
        "majors": majors,
    }

# ---------------------------------------------------------------------------
# Past-event helper
# ---------------------------------------------------------------------------

def get_past_event_results(event_name: str) -> list[dict]:
    field = tournament_fields.get(event_name)
    if not field:
        return []
    player_odds = load_event_odds(event_name, str(DATA_DIR))
    rows = []
    for player, info in field.items():
        row = {
            "PlayerName": player,
            "FinishingPosition": info.get("Finishing Position", 999),
        }
        odds = player_odds.get(player, {})
        for col in ("WinOdds", "T5Odds", "T10Odds", "T20Odds"):
            row[col] = odds.get(col)
        rows.append(row)
    rows.sort(key=lambda r: r["FinishingPosition"])
    return rows

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="Golf Trends Dashboard")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


CURRENT_NEXT_EVENT = max(NEXT_EVENTS, key=NEXT_EVENTS.get)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    past = sorted(
        (e for e in NEXT_EVENTS if e != CURRENT_NEXT_EVENT),
        key=lambda e: NEXT_EVENTS[e],
        reverse=True,
    )
    return templates.TemplateResponse("index.html", {
        "request": request,
        "next_event": CURRENT_NEXT_EVENT,
        "past_events": past,
        "players": sorted(owgr_dict.keys()),
    })


@app.get("/api/next-event/{event_name}")
async def api_next_event(event_name: str):
    return get_event_predictions(event_name)


@app.get("/api/player/{player_name}")
async def api_player(player_name: str):
    return get_player_timeseries(player_name)


@app.get("/api/past-event/{event_name}")
async def api_past_event(event_name: str):
    return get_past_event_results(event_name)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
