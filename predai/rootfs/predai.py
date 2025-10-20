#!/usr/bin/env python3
"""
PredAI (fork) — config‑driven multi‑horizon forecasting for Home Assistant.

Key features:
* Configurable sensor roles (energy counters, temperature, Mixergy immersion demand, etc.).
* Automatic power→energy integration for non‑cumulative power sensors (mean power -> kWh/interval).
* Covariate alias + scaling from YAML covariates: section.
* Interval, cumulative, daily_cum, and horizon (+2h/+8h/+12h or custom) forecast publishing.
* Async Home Assistant API via aiohttp.
* SQLite history cache.
* Config hot‑reload each cycle.

British spelling used in comments.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import json

import numpy as np
import pandas as pd
import yaml

# NeuralProphet
try:
    from neuralprophet import NeuralProphet, set_log_level as np_set_log_level
except Exception as e:  # pragma: no cover
    NeuralProphet = None
    _NP_IMPORT_ERROR = e
else:
    _NP_IMPORT_ERROR = None
    np_set_log_level("ERROR")

import aiohttp
import aiohttp.client_exceptions

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

TIMEOUT = 240
TIME_FORMAT_HA = "%Y-%m-%dT%H:%M:%S%z"
TIME_FORMAT_HA_DOT = "%Y-%m-%dT%H:%M:%S.%f%z"

DEFAULT_CONFIG_PATH = "/config/predai.yaml"
DEFAULT_DB_PATH = "/config/predai.db"

DEFAULT_PUBLISH_PREFIX = "predai_"
DEFAULT_INTERVAL_MIN = 30
DEFAULT_HORIZONS_MIN = [120, 480, 720]  # +2h, +8h, +12h

SAFE_TBL_RE = re.compile(r"^[A-Za-z0-9_]+$")

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logger = logging.getLogger("predai")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(logging.DEBUG)
# ---- PredAI logging helpers ----
def summarise_series(name: str, s: pd.Series):
    try:
        size = int(s.shape[0])
        nans = int(s.isna().sum())
        s_clean = s.dropna()
        vmin = float(s_clean.min()) if not s_clean.empty else None
        vmax = float(s_clean.max()) if not s_clean.empty else None
        logger.info("Summary %s: size=%s NaNs=%s min=%s max=%s", name, size, nans, vmin, vmax)
    except Exception as e:
        logger.warning("Summary failed for %s: %s", name, e)

def summarise_df(tag: str, df: pd.DataFrame):
    try:
        logger.info("DF %s: shape=%s cols=%s", tag, df.shape, list(df.columns))
        if "ds" in df.columns and not df.empty:
            logger.info("DF %s time span: %s -> %s", tag, df["ds"].min(), df["ds"].max())
        for c in df.columns:
            if c == "ds":
                continue
            summarise_series(f"{tag}.{c}", df[c])
    except Exception as e:
        logger.warning("DF summary failed for %s: %s", tag, e)
# ---- end helpers ----


# --- Gap-aware cumulative → interval conversion ---

def _cum_to_interval_gap_aware_single_day(s: pd.Series, freq: str) -> pd.Series:
    """
    s: cumulative values for a single day (sorted, tz-aware or naive)
    Returns interval increments on a regular freq grid, distributing gaps.
    """
    if s is None or s.empty:
        return pd.Series(dtype="float32")
    # keep finite & sorted
    s = s[pd.notna(s)].sort_index()
    if s.empty:
        return pd.Series(dtype="float32")
    # Interpolate cumulative to the regular grid, then edge-diff
    grid = s.resample(freq).interpolate("time")  # linear interpolation over gaps
    y = grid.diff().fillna(0.0).clip(lower=0.0)
    return y.astype("float32")


def cum_to_interval_gap_aware(cum: pd.Series, freq: str, reset_daily: bool = False, tz: str | None = None) -> pd.Series:
    """
    Convert cumulative series to interval series on a regular grid.
    If reset_daily=True, operate within each local day separately.
    """
    if cum is None or len(cum) == 0:
        # still return an empty grid (caller typically resamples later)
        return pd.Series(dtype="float32")

    s = cum[pd.notna(cum)].sort_index()
    if s.empty:
        return pd.Series(dtype="float32")

    # Optional: enforce non-decreasing within a day (comment out if not desired)
    # s = s.cummax()

    if not reset_daily:
        return _cum_to_interval_gap_aware_single_day(s, freq)

    # Daily mode: split by local day, interpolate per-day, then concat
    idx = s.index
    if getattr(idx, "tz", None) is not None and tz is not None:
        day_keys = idx.tz_convert(tz).normalize()
    else:
        # fall back to existing timezone or naive midnights
        day_keys = idx.normalize()

    parts = []
    for _, part in s.groupby(day_keys, sort=True):
        y_part = _cum_to_interval_gap_aware_single_day(part, freq)
        parts.append(y_part)
    if not parts:
        return pd.Series(dtype="float32")
    y = pd.concat(parts).sort_index()
    return y.astype("float32")


def clip_outliers_quantile(s, q: float = 0.995, positive_only: bool = True):
    """
    Clip the upper tail at the q-quantile.
    If positive_only is True, compute the quantile using only strictly-positive values.
    If the computed cap is <= 0 or cannot be computed, skip clipping.
    """
    yy = pd.Series(s).astype("float32")
    ref = yy[yy > 0] if positive_only else yy

    if ref.empty:
        # Nothing to measure; skip
        return yy

    cap = ref.quantile(q)
    if cap is None or not np.isfinite(cap) or float(cap) <= 0.0:
        # Unsafe cap; skip
        return yy

    return yy.clip(lower=0.0, upper=float(cap)).astype("float32")


def _get_local_tz(cfg):
    """Return configured timezone string or 'UTC'."""
    try:
        # If cfg already carries a timezone string, reuse it
        tz = getattr(cfg, "timezone", None)
        if tz:
            return tz
        tzname = getattr(cfg, "timezone_name", None)
        if tzname:
            return tzname
        tzobj = getattr(cfg, "tz", None)
        if tzobj:
            return str(tzobj)
    except Exception:
        pass
    return "UTC"


def is_daily_cumulative_sensor(entity_id: str, unit: str | None, state_class: str | None) -> bool:
    """Heuristic: identify daily cumulative sensors (e.g., names containing 'today' or 'daily' + total_increasing)."""
    name = (entity_id or "").lower()
    unit = (unit or "").lower()
    sc = (state_class or "").lower()
    return (("today" in name) or ("daily" in name)) and ("total_increasing" in sc)


def cumulative_to_interval_daily(raw_df: pd.DataFrame,
                                 ts_col: str = "ts",
                                 val_col: str = "val",
                                 local_tz: str = "UTC",
                                 drop_negatives: bool = True,
                                 tiny_eps: float = 1e-6) -> pd.DataFrame:
    """
    Convert a daily-reset total_increasing series to intervals by:
      1) grouping by local calendar day,
      2) diff within each day,
      3) clipping tiny/negative deltas to 0,
      4) first sample of each day becomes 0 (diff=NaN -> 0).
    Returns a DataFrame with columns [ts, y] at original timestamps (not yet resampled).
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=[ts_col, "y"])

    df = raw_df[[ts_col, val_col]].dropna().copy()
    if df.empty:
        return pd.DataFrame(columns=[ts_col, "y"])

    # Ensure tz-aware timestamps
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    # Convert to local tz to compute local-day grouping
    local = df[ts_col].dt.tz_convert(local_tz)
    df["_day"] = local.dt.floor("D")

    # Diff within each local day
    df["y"] = df.groupby("_day", sort=False)[val_col].diff()

    # Handle negatives/noise: tiny negatives and any negatives -> 0
    neg = df["y"] < 0
    tiny = df["y"].between(-tiny_eps, 0, inclusive="neither")
    df.loc[neg | tiny, "y"] = 0.0

    # First value per day had NaN diff → 0
    df["y"] = df["y"].fillna(0.0)

    out = df[[ts_col, "y"]].copy()
    out["y"] = out["y"].astype("float32")
    return out


def resample_intervals_sum(df: pd.DataFrame,
                           ts_col: str = "ts",
                           y_col: str = "y",
                           freq: str = "30min") -> pd.DataFrame:
    """
    Resample an interval series to fixed cadence by sum.
    Missing bins become 0 (no forward fill).
    Returns columns ['ds','y'].
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["ds", "y"])
    x = df.set_index(ts_col)[y_col].astype("float32")
    y = x.resample(freq).sum().asfreq(freq, fill_value=0.0).astype("float32")
    return y.reset_index().rename(columns={ts_col: "ds", y_col: "y"})


def cumulative_to_interval_naive_fallback(raw_df: pd.DataFrame,
                                          resample_freq: str = "30min") -> pd.DataFrame:
    """
    Prior behavior: simple diff across entire series, negatives -> 0, then resample by sum.
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=["ds", "y"])
    df = raw_df[["ts", "val"]].dropna().copy()
    if df.empty:
        return pd.DataFrame(columns=["ds", "y"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts")
    diff = df["val"].astype("float32").diff().fillna(0.0)
    diff = diff.mask(diff < 0.0, 0.0).astype("float32")
    intervals = pd.DataFrame({"ts": df["ts"], "y": diff})
    y30 = resample_intervals_sum(intervals, ts_col="ts", y_col="y", freq=resample_freq)
    pos_after = int((y30["y"] > 0).sum())
    nonzero_share = 100.0 * pos_after / max(1, len(y30))
    logging.info("Naive diff result rows=%d; positive buckets=%d (%.2f%%); y.max=%.3f; y.sum=%.3f",
                 len(y30), pos_after, nonzero_share, float(y30["y"].max()), float(y30["y"].sum()))
    return y30


def cumulative_to_interval_gapaware_with_daily(raw_df: pd.DataFrame,
                                               entity_id: str,
                                               unit: str | None,
                                               state_class: str | None,
                                               cfg,
                                               resample_freq: str = "30min") -> pd.DataFrame:
    """
    Preferred path for sensors like '*…today*' with total_increasing semantics:
      1) daily segmentation diff → intervals
      2) resample by sum to fixed cadence
      3) if result is all zeros, fall back to naive diff+resample (with warning)
    """
    tz = _get_local_tz(cfg)
    # Decide whether to try daily segmentation first
    if is_daily_cumulative_sensor(entity_id, unit, state_class):
        logging.debug("Cumulative->interval(daily) on %d rows (entity=%s, tz=%s)",
                      0 if raw_df is None else len(raw_df), entity_id, tz)
        intervals = cumulative_to_interval_daily(raw_df, ts_col="ts", val_col="val", local_tz=tz)
        pos_before = int((intervals["y"] > 0).sum()) if not intervals.empty else 0
        logging.debug("Daily-seg: positives before resample=%d / %d", pos_before, 0 if intervals is None else len(intervals))

        y30 = resample_intervals_sum(intervals, ts_col="ts", y_col="y", freq=resample_freq)
        pos_after = int((y30["y"] > 0).sum()) if not y30.empty else 0
        nonzero_share = 100.0 * pos_after / max(1, len(y30))
        logging.info("Daily-seg result rows=%d; positive buckets=%d (%.2f%%); y.max=%.3f; y.sum=%.3f",
                     len(y30), pos_after, nonzero_share,
                     float(0 if y30.empty else y30["y"].max()),
                     float(0 if y30.empty else y30["y"].sum()))

        if pos_after == 0:
            logging.warning("Daily-seg produced all zeros; falling back to naive diff.")
            return cumulative_to_interval_naive_fallback(raw_df, resample_freq=resample_freq)
        return y30

    # Otherwise, use the naive path directly
    return cumulative_to_interval_naive_fallback(raw_df, resample_freq=resample_freq)


# --------------------------------------------------------------------------- #
# Utility: timestamps
# --------------------------------------------------------------------------- #

def timestr_to_datetime(timestamp: str) -> Optional[datetime]:
    """Parse a Home‑Assistant timestamp and return a tz‑aware datetime
    rounded to the nearest minute (seconds & microseconds cleared).

    Accepts:
      • ISO‑8601 with colon in offset, e.g. '2025‑07‑28T10:47:12+01:00'
      • ISO‑8601 with fractional seconds
      • Legacy HA formats without the colon
    """
    if not timestamp:
        return None

    # 1.  Robust ISO parser (handles “+01:00”)
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.replace(second=0, microsecond=0)
    except ValueError:
        pass  # fall back to legacy formats

    # 2.  Legacy formats
    for fmt in (TIME_FORMAT_HA, TIME_FORMAT_HA_DOT):
        try:
            dt = datetime.strptime(timestamp, fmt)
            return dt.replace(second=0, microsecond=0)
        except ValueError:
            continue

    return None


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# --------------------------------------------------------------------------- #
# Config dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class RoleCfg:
    target_transform: str = "interval"      # interval|level|cumulative
    aggregation: str = "sum"                # sum|mean|last
    publish_state_class: str = "measurement"
    model_backend: str = "neuralprophet"
    n_lags: int = 8
    seasonality_reg: float = 5.0
    seasonality_mode: str = "additive"
    learning_rate: Optional[float] = None


@dataclass
class ResetDetectionCfg:
    enabled: bool = False
    low: float = 1.0
    high: float = 2.0
    hard_reset_value: Optional[float] = None


@dataclass
class SensorCfg:
    name: str
    role: str
    units: str = ""
    output_units: Optional[str] = None        # override publish units (e.g., W in, kWh out)
    days_hist: int = 7
    export_days: Optional[int] = None

    subtract: List[str] = field(default_factory=list)
    incrementing: bool = False
    reset_daily: bool = False
    interval: Optional[int] = None
    future_periods: Optional[int] = None
    reset_low: Optional[float] = None
    reset_high: Optional[float] = None

    source_is_cumulative: bool = False
    train_target: str = "interval"            # interval|level|cumulative
    aggregation: Optional[str] = None         # resample override
    log_transform: bool = False

    reset_detection: ResetDetectionCfg = field(default_factory=ResetDetectionCfg)

    publish_interval: bool = True
    publish_cumulative: bool = True
    publish_daily_cumulative: bool = True

    covariates_future: List[str] = field(default_factory=list)
    covariates_lagged: List[str] = field(default_factory=list)
    covariates_both: List[str] = field(default_factory=list)
    covariates_binary: Optional[List[str]] = None  # names explicitly treated as binary
    binary_lag_cap: Optional[int] = 96            # cap n_lags for binary covariates (None = no cap)

    n_lags: Optional[int] = None
    seasonality_reg: Optional[float] = None
    seasonality_mode: Optional[str] = None
    learning_rate: Optional[float] = None
    country: Optional[str] = None

    database: bool = True
    max_age: int = 365
    max_increment: Optional[float] = None

    plot: bool = False
    cascade_outputs: Dict[str, bool] = field(default_factory=dict)
    publish_name: Optional[str] = None

    def effective_aggregation(self, role_cfg: RoleCfg) -> str:
        return self.aggregation or role_cfg.aggregation

    def effective_n_lags(self, role_cfg: RoleCfg) -> int:
        return self.n_lags if self.n_lags is not None else role_cfg.n_lags

    def effective_seasonality_reg(self, role_cfg: RoleCfg) -> float:
        return self.seasonality_reg if self.seasonality_reg is not None else role_cfg.seasonality_reg

    def effective_seasonality_mode(self, role_cfg: RoleCfg) -> str:
        return self.seasonality_mode or role_cfg.seasonality_mode

    def effective_learning_rate(self, role_cfg: RoleCfg) -> Optional[float]:
        return self.learning_rate if self.learning_rate is not None else role_cfg.learning_rate


@dataclass
class PredAIConfig:
    update_every: int = 30
    common_interval: int = DEFAULT_INTERVAL_MIN
    horizons: List[int] = field(default_factory=lambda: DEFAULT_HORIZONS_MIN)
    publish_prefix: str = DEFAULT_PUBLISH_PREFIX
    defaults: Dict[str, Any] = field(default_factory=dict)
    roles: Dict[str, RoleCfg] = field(default_factory=dict)
    sensors: List[SensorCfg] = field(default_factory=list)
    timezone_name: str = "Europe/London"
    cov_map: Dict[str, Any] = field(default_factory=dict)

    @property
    def tz(self):
        try:
            from zoneinfo import ZoneInfo
            return ZoneInfo(self.timezone_name)
        except Exception:
            logger.warning("Could not load timezone %s; falling back to UTC.", self.timezone_name)
            return timezone.utc


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #

def _load_role(name: str, d: Dict[str, Any]) -> RoleCfg:
    return RoleCfg(
        target_transform=d.get("target_transform", "interval"),
        aggregation=d.get("aggregation", "sum"),
        publish_state_class=d.get("publish_state_class", "measurement"),
        model_backend=d.get("model", {}).get("backend", "neuralprophet"),
        n_lags=d.get("model", {}).get("n_lags", 8),
        seasonality_reg=d.get("model", {}).get("seasonality_reg", 5.0),
        seasonality_mode=d.get("model", {}).get("seasonality_mode", "additive"),
        learning_rate=d.get("model", {}).get("learning_rate"),
    )


def _load_sensor(dflt: Dict[str, Any], d: Dict[str, Any]) -> SensorCfg:
    merged = dict(dflt)
    merged.update(d)
    rd_map = merged.get("reset_detection", {})
    rd = ResetDetectionCfg(
        enabled=rd_map.get("enabled", False),
        low=rd_map.get("low", 1.0),
        high=rd_map.get("high", 2.0),
        hard_reset_value=rd_map.get("hard_reset_value"),
    )
    subtract_val = merged.get("subtract", [])
    if isinstance(subtract_val, str):
        subtract_list = [subtract_val]
    else:
        subtract_list = list(subtract_val) if subtract_val else []

    incrementing_val = merged.get("incrementing")
    if incrementing_val is not None:
        merged.setdefault("source_is_cumulative", bool(incrementing_val))

    if merged.get("reset_low") is not None:
        rd.low = float(merged["reset_low"])
    if merged.get("reset_high") is not None:
        rd.high = float(merged["reset_high"])
    return SensorCfg(
        name=merged["name"],
        role=merged.get("role", "incrementing_energy"),
        units=merged.get("units", ""),
        output_units=merged.get("output_units"),
        days_hist=merged.get("days", merged.get("days_hist", 7)),
        export_days=merged.get("export_days"),
        subtract=subtract_list,
        incrementing=bool(incrementing_val) if incrementing_val is not None else False,
        reset_daily=merged.get("reset_daily", False),
        interval=merged.get("interval"),
        future_periods=merged.get("future_periods"),
        reset_low=merged.get("reset_low"),
        reset_high=merged.get("reset_high"),
        source_is_cumulative=merged.get("source_is_cumulative", False),
        train_target=merged.get("train_target", "interval"),
        aggregation=merged.get("aggregation"),
        log_transform=merged.get("log_transform", False),
        reset_detection=rd,
        publish_interval=merged.get("publish_interval", True),
        publish_cumulative=merged.get("publish_cumulative", True),
        publish_daily_cumulative=merged.get("publish_daily_cumulative", True),
        covariates_future=(merged.get("covariates_future") or merged.get("future") or []),
        covariates_lagged=(merged.get("covariates_lagged") or merged.get("lagged") or []),
        covariates_both=(merged.get("covariates_both") or merged.get("lagged_future") or []),
        covariates_binary=merged.get("covariates_binary"),
        binary_lag_cap=merged.get("binary_lag_cap", 96),
        n_lags=merged.get("n_lags"),
        seasonality_reg=merged.get("seasonality_reg"),
        seasonality_mode=merged.get("seasonality_mode"),
        learning_rate=merged.get("learning_rate"),
        country=merged.get("country"),
        database=merged.get("database", True),
        max_age=merged.get("max_age", 365),
        max_increment=merged.get("max_increment"),
        plot=merged.get("plot", False),
        cascade_outputs=merged.get("cascade_outputs", {}) or {},
        publish_name=merged.get("publish_name"),
    )


def load_config(path: str = DEFAULT_CONFIG_PATH) -> PredAIConfig:
    """Load YAML configuration for PredAI.

    A short info log is emitted with the path being loaded so users can
    verify which file is in use. If the file is missing, the function
    falls back to defaults and reports an error.
    """
    logger.info("Loading configuration from %s", path)
    if not os.path.exists(path):
        logger.error("Configuration file %s not found.", path)
        return PredAIConfig()
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    dflt = raw.get("defaults", {})
    publish_prefix = dflt.get("publish_prefix", DEFAULT_PUBLISH_PREFIX)

    roles_raw = raw.get("roles", {}) or {}
    roles = {k: _load_role(k, v) for k, v in roles_raw.items()}

    sensors_raw = raw.get("sensors", []) or []
    sensors: List[SensorCfg] = []
    for s in sensors_raw:
        s_clean = {k: v for k, v in s.items() if k != "roles"}  # ignore stray nested roles
        sensors.append(_load_sensor(dflt, s_clean))

    logger.info("Configuration: %s roles, %s sensors", len(roles), len(sensors))

    cfg = PredAIConfig(
        update_every=raw.get("update_every", 30),
        common_interval=raw.get("common_interval", DEFAULT_INTERVAL_MIN),
        horizons=raw.get("horizons", DEFAULT_HORIZONS_MIN),
        publish_prefix=publish_prefix,
        defaults=dflt,
        roles=roles,
        sensors=sensors,
        timezone_name=raw.get("timezone", "Europe/London"),
        cov_map=raw.get("covariates", {}) or {},
    )
    return cfg


# --------------------------------------------------------------------------- #
# Home Assistant Interface
# --------------------------------------------------------------------------- #

class HAInterface:
    def __init__(self, ha_url: Optional[str], ha_key: Optional[str], session: Optional[aiohttp.ClientSession] = None):
        self.ha_url = ha_url or "http://192.168.188.117"
        self.ha_key = ha_key or os.environ.get("SUPERVISOR_TOKEN")
        if not self.ha_key:
            raise RuntimeError("No Home Assistant key found.")
        self._session = session
        mask = self.ha_key[:6] + "…" if len(self.ha_key) >= 6 else "***"
        logger.info("HA Interface initialised (token %s, url %s)", mask, self.ha_url)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=TIMEOUT)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def api_call(self, method: str, endpoint: str, params: Optional[dict] = None, json_data: Optional[dict] = None) -> Any:
        url = self.ha_url.rstrip("/") + endpoint
        sess = await self._get_session()
        headers = {
            "Authorization": f"Bearer {self.ha_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            async with sess.request(method, url, headers=headers, params=params, json=json_data) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    logger.warning("HA API %s %s -> %s: %s", method, endpoint, resp.status, text)
                try:
                    return json.loads(text) if text else None
                except json.JSONDecodeError:
                    logger.error("Non‑JSON response from %s: %s", url, text)
                    return None
        except aiohttp.client_exceptions.ClientError as e:
            logger.error("HA API error %s %s: %s", method, endpoint, e)
            return None

    async def get_history(self, sensor: str, start: datetime, end: datetime) -> Tuple[List[dict], Optional[datetime], Optional[datetime]]:
        params = {
            "filter_entity_id": sensor,
            "end_time": end.strftime(TIME_FORMAT_HA),
        }
        endpoint = "/api/history/period/" + start.strftime(TIME_FORMAT_HA)
        res = await self.api_call("GET", endpoint, params=params)
        if not res:
            logger.warning("No history for %s", sensor)
            return [], None, None
        arr = res[0] if isinstance(res, list) and res else []
        try:
            st = timestr_to_datetime(arr[0]["last_updated"]) if arr else None
            en = timestr_to_datetime(arr[-1]["last_updated"]) if arr else None
        except Exception:
            st = en = None
        return arr, st, en

    async def get_state(self, entity_id: str, default: Any = None, attribute: Optional[str] = None):
        item = await self.api_call("GET", f"/api/states/{entity_id}")
        if not item:
            return default
        if attribute:
            return item.get("attributes", {}).get(attribute, default)
        return item.get("state", default)

    async def set_state(self, entity_id: str, state: Any, attributes: Optional[dict] = None):
        data = {"state": str(state)}
        if attributes:
            data["attributes"] = attributes
        await self.api_call("POST", f"/api/states/{entity_id}", json_data=data)


# --------------------------------------------------------------------------- #
# SQLite history cache
# --------------------------------------------------------------------------- #

class HistoryDB:
    def __init__(self, path: str = DEFAULT_DB_PATH):
        self.path = path
        self.con = sqlite3.connect(self.path)
        self.cur = self.con.cursor()

    def safe_name(self, name: str) -> str:
        t = name.replace(".", "_")
        if not SAFE_TBL_RE.match(t):
            raise ValueError(f"Unsafe table name: {name}")
        return t

    def create_table(self, table: str):
        t = self.safe_name(table)
        self.cur.execute(f"CREATE TABLE IF NOT EXISTS {t} (timestamp TEXT PRIMARY KEY, value REAL)")
        self.con.commit()

    def cleanup_table(self, table: str, oldest_dt: datetime):
        t = self.safe_name(table)
        oldest_stamp = oldest_dt.strftime("%Y-%m-%d %H:%M:%S%z")
        self.cur.execute(f"DELETE FROM {t} WHERE timestamp < ?", (oldest_stamp,))
        self.con.commit()

    def get_history(self, table: str) -> pd.DataFrame:
        t = self.safe_name(table)
        # Ensure the table exists so the SELECT does not fail on first run
        self.create_table(t)
        self.cur.execute(f"SELECT * FROM {t} ORDER BY timestamp")
        rows = self.cur.fetchall()
        if not rows:
            return pd.DataFrame(columns=["ds", "y"])
        df = pd.DataFrame(rows, columns=["ds", "y"])
        df["ds"] = pd.to_datetime(df["ds"], utc=True, errors="coerce")
        return df.dropna(subset=["ds"])

    def store_history(self, table: str, history: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
        t = self.safe_name(table)
        self.create_table(t)
        # Normalise previously stored timestamps to the same ISO format that we
        # use when inserting new rows. `str(dt) would produce a space between
        # date and time ("YYYY-MM-DD HH:MM:SS+00:00"), whereas `isoformat()
        # yields "YYYY-MM-DDTHH:MM:SS+00:00".  The mismatch allowed duplicates to
        # slip past the "timestamp_s not in prev_values" check and triggered
        # SQLite UNIQUE constraint errors.  Build the set using `isoformat so
        # comparisons are consistent.
        prev_values = set(dt.isoformat() for dt in prev["ds"] if pd.notna(dt))
        added = 0
        for _, row in history.iterrows():
            timestamp = pd.to_datetime(row["ds"], utc=True, errors="coerce")
            if pd.isna(timestamp):
                continue
            timestamp_s = timestamp.isoformat()
            value = float(row["y"])
            if timestamp_s not in prev_values:
                self.cur.execute(
                    f"INSERT OR IGNORE INTO {t} (timestamp, value) VALUES (?, ?)",
                    (timestamp_s, value),
                )
                if self.cur.rowcount:
                    prev_values.add(timestamp_s)
                    prev.loc[len(prev)] = {"ds": timestamp, "y": value}
                    added += 1
        self.con.commit()
        logger.info("DB: added %s rows to %s", added, t)
        return prev

    def close(self):
        self.con.close()


# --------------------------------------------------------------------------- #
# Transform utilities
# --------------------------------------------------------------------------- #

def _state_to_float(x) -> float | None:
    # Handle real booleans fast
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    if x is None:
        return None

    s = str(x).strip().lower()
    # Explicit NA-ish states from HA
    if s in {"unknown", "unavailable", "none", "nan"}:
        return None

    # Common boolean-ish labels
    truthy = {"on", "true", "open", "home", "detected", "motion", "active", "present"}
    falsy  = {"off", "false", "closed", "not_home", "clear", "no_motion", "inactive", "absent"}

    if s in truthy:
        return 1.0
    if s in falsy:
        return 0.0

    # Try numeric strings (e.g. "0", "1")
    try:
        return float(s)
    except Exception:
        return None
    

def _parse_ts_any(x):
    """Return a UTC pandas.Timestamp or NaT for a single value."""
    # Fast paths for common types
    if isinstance(x, pd.Timestamp):
        return x.tz_convert("UTC") if x.tzinfo is not None else x.tz_localize("UTC")
    if isinstance(x, datetime):
        return pd.Timestamp(x.astimezone(timezone.utc)) if x.tzinfo else pd.Timestamp(x, tz="UTC")
    # Numeric epoch seconds (or numeric-looking string)
    try:
        # cheap numeric check
        if isinstance(x, (int, float, np.integer, np.floating)) or (isinstance(x, str) and x.replace(".","",1).lstrip("-").isdigit()):
            return pd.to_datetime(x, unit="s", utc=True, errors="coerce")
    except Exception:
        pass
    # Fallback: parse as string (covers ISO8601, with/without Z/offset/micros)
    try:
        return pd.to_datetime(str(x), utc=True, errors="coerce")
    except Exception:
        return pd.NaT

def normalise_history(raw: list[dict]) -> pd.DataFrame:
    logger.debug("Normalising history with %s raw rows", len(raw))
    if not raw:
        return pd.DataFrame(columns=["ds", "value"])

    df = pd.DataFrame(raw)
    logger.debug("History columns: %s", list(df.columns))

    # --- Build robust UTC timestamp per row from any of the known fields ---
    # Collect candidates (whatever exists), then row-wise coalesce first non-NaT
    candidates = []
    for col, kind in [
        ("last_updated", "iso"),
        ("last_changed", "iso"),
        ("last_updated_ts", "epoch"),
        ("last_changed_ts", "epoch"),
    ]:
        if col in df.columns:
            if kind == "epoch":
                s = pd.to_datetime(df[col], unit="s", utc=True, errors="coerce")
            else:
                # apply handles mixed objects/strings/datetimes safely
                s = df[col].apply(_parse_ts_any)
            candidates.append(s)

    if not candidates:
        logger.warning("History has no recognised timestamp columns.")
        return pd.DataFrame(columns=["ds", "value"])

    ds = candidates[0]
    for s in candidates[1:]:
        ds = ds.fillna(s)
    df["ds"] = ds

    # --- State → numeric (binary map, numeric-with-units tolerant) ---
    coerced = df["state"].map(_state_to_float)  # your existing helper
    if coerced.isna().any():
        num_token = (
            df.loc[coerced.isna(), "state"]
              .astype(str)
              .str.extract(r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", expand=False)
        )
        coerced = coerced.where(coerced.notna(), pd.to_numeric(num_token, errors="coerce"))
    numeric = pd.to_numeric(df["state"], errors="coerce")
    df["value"] = coerced.where(coerced.notna(), numeric).astype("float32")

    # --- Clean, order, dedupe ---
    n0 = len(df)
    miss_ds = int(df["ds"].isna().sum())
    miss_val = int(df["value"].isna().sum())
    # (optional) peek a few problematic rows to debug formats
    bad_sample = df.loc[df["ds"].isna(), ["last_updated","last_changed"]].head(3).to_dict("records")
    if bad_sample:
        logger.debug("Sample rows with unparseable timestamps: %s", bad_sample)

    df = df.dropna(subset=["ds", "value"]).sort_values("ds")
    df = df.drop_duplicates(subset=["ds"], keep="last")

    logger.debug(
        "Normalised history: rows=%s (raw=%s, missing ds=%s, missing value=%s)",
        len(df), n0, miss_ds, miss_val
    )
    return df[["ds", "value"]]


def _complete_grid_after_resample(df: pd.DataFrame, freq: str, how: str, tz=None, ds_col: str = "ds", val_col: str = "y") -> pd.DataFrame:
    """
    Ensure a regular time grid after resample and fill gaps deterministically.

    - For interval-like series (how == "sum"): fill missing bins with 0.0.
    - For level-like series (how != "sum"): forward-fill, then back-fill one step.
    Returns a frame with columns [ds_col, val_col] on a complete grid [start..end] with cadence=freq.
    """
    assert ds_col in df.columns and val_col in df.columns, (df.columns, ds_col, val_col)
    ser = df.set_index(ds_col)[val_col]
    tz = getattr(df[ds_col].dt, "tz", None) or tz
    start = df[ds_col].min()
    end = df[ds_col].max()
    full_idx = pd.date_range(start=start, end=end, freq=freq, tz=tz)
    ser = ser.reindex(full_idx)

    if how == "sum":
        ser = ser.fillna(0.0)
    else:
        ser = ser.ffill().bfill(limit=1)

    out = ser.rename(val_col).to_frame().reset_index().rename(columns={"index": ds_col})
    return out


def resample_sensor(df: pd.DataFrame, freq: str, how: str) -> pd.DataFrame:
    if df.empty:
        return df
    logger.debug(
        "Resampling %s rows to freq=%s how=%s", len(df), freq, how
    )
    df = df.set_index("ds").sort_index()
    if how == "sum":
        agg = df["value"].resample(freq).sum(min_count=1)
    elif how == "last":
        agg = df["value"].resample(freq).last()
    else:  # mean default
        agg = df["value"].resample(freq).mean()
    out = agg.to_frame("value").reset_index()
    out = _complete_grid_after_resample(out, freq=freq, how=how, ds_col="ds", val_col="value")
    logger.debug("Resampled result rows=%s", len(out))
    return out


# --- Binary covariate utilities ---------------------------------------------
BOOL_STR_MAP = {
    "on": 1.0, "off": 0.0,
    "true": 1.0, "false": 0.0,
    "yes": 1.0, "no": 0.0,
    "open": 1.0, "closed": 0.0,
    "home": 1.0, "away": 0.0,
    "detected": 1.0, "clear": 0.0,
    "armed": 1.0, "disarmed": 0.0,
}

def _to_numeric_or_bool(s: pd.Series) -> pd.Series:
    """Map common boolean-ish strings to 0/1, else numeric; keep NaN."""
    if s.dtype == object:
        s2 = s.astype(str).str.strip().str.lower().map(BOOL_STR_MAP)
        num = pd.to_numeric(s, errors="coerce")
        s = s2.where(s2.notna(), num)
    else:
        s = pd.to_numeric(s, errors="coerce")
    return s

def is_binary_like(series: pd.Series, min_obs: int = 20) -> bool:
    """True if values (after coercion) are subset of {0,1}."""
    x = _to_numeric_or_bool(series).dropna().astype(float)
    if len(x) == 0:
        return False
    uniq = pd.unique(x.round(6))
    return set(np.round(uniq, 6)).issubset({0.0, 1.0})

def coerce_to_binary(series: pd.Series) -> pd.Series:
    """Force strict 0/1 without NaNs (ffill/bfill/0)."""
    x = _to_numeric_or_bool(series)
    if x.dropna().between(0.0, 1.0).all():
        x = (x.fillna(method="ffill").fillna(method="bfill").fillna(0.0) > 0.5).astype(float)
    else:
        x = x.round().clip(0, 1).astype(float)
    return x

def resample_covariate(series: pd.Series, freq: str, *, is_binary: bool) -> pd.Series:
    """Resample: binary uses step (ffill); continuous uses mean."""
    s = series.sort_index()
    if is_binary:
        s = coerce_to_binary(s)
        s = s.resample(freq).ffill()
        s = s.ffill().bfill().fillna(0.0)
        s = s.clip(0, 1).round().astype(float)
    else:
        s = s.resample(freq).mean()
    return s

def is_binary_cov_name(name: str, sensor_cfg) -> bool:
    cfg_list = set(sensor_cfg.covariates_binary or [])
    return name in cfg_list

def _lags_for_covariate(cov_name: str, base_n_lags: int, sensor_cfg) -> int:
    """Optionally cap lags for binary covariates."""
    if is_binary_cov_name(cov_name, sensor_cfg):
        cap = sensor_cfg.binary_lag_cap
        return min(base_n_lags, cap) if cap else base_n_lags
    return base_n_lags


def cumulative_to_interval(df: pd.DataFrame, reset_cfg: ResetDetectionCfg, max_increment: Optional[float] = None) -> pd.DataFrame:
    if df.empty:
        df["y"] = []
        return df
    logger.debug("Cumulative->interval on %s rows", len(df))
    df = df.sort_values("ds").reset_index(drop=True)
    v = df["value"].to_numpy()
    delta = np.diff(v, prepend=v[0])
    delta[0] = np.nan
    neg_mask = delta < 0
    reset_count = 0
    if reset_cfg.enabled:
        if reset_cfg.hard_reset_value is not None:
            reset_mask = np.isclose(v, reset_cfg.hard_reset_value)
            reset_count = int(np.sum(reset_mask))
            for i in np.where(reset_mask)[0]:
                delta[i] = v[i]
    delta[neg_mask] = np.nan
    spike_count = 0
    if max_increment is not None:
        spike_mask = np.abs(delta) > max_increment
        spike_count = int(np.sum(spike_mask))
        delta[spike_mask] = np.nan
    delta = np.nan_to_num(delta, nan=0.0)
    delta = np.clip(delta, 0.0, None)
    df["y"] = delta
    logger.debug(
        "Cumulative->interval result rows=%s neg=%s resets=%s spikes=%s",
        len(df),
        int(neg_mask.sum()),
        reset_count,
        spike_count,
    )
    return df


def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Applying log transform to %s rows", len(df))
    df = df.copy()
    df["y"] = np.log1p(df["y"].clip(lower=0))
    df.attrs["log_transform_applied"] = True
    logger.debug("Log transform complete")
    return df


def invert_log_transform(arr: np.ndarray, applied: bool) -> np.ndarray:
    if not applied:
        return arr
    return np.expm1(arr)


def subtract_set(base: pd.DataFrame, sub: pd.DataFrame, *, inc: bool = False) -> pd.DataFrame:
    """Subtract one dataset from another by timestamp."""
    if base.empty:
        return base
    merged = base.merge(sub, on="ds", how="left", suffixes=("", "_sub"))
    merged["y_sub"].fillna(0, inplace=True)
    merged["y"] = merged.apply(
        lambda row: max(row["y"] - row["y_sub"], 0) if inc else row["y"] - row["y_sub"],
        axis=1,
    )
    return merged[["ds", "y"]]

def resolve_n_lags(sensor_cfg, role_cfg, train_rows: int, n_forecasts: int) -> int:
    """
    Returns the final integer n_lags.
    - If sensor/model config says 'auto', choose the maximum permissible value
      so there's at least one training window: len(train_df) - n_forecasts - 1.
    - Otherwise, coerce to int.
    """
    # Prefer sensor override, else role default
    raw = sensor_cfg.n_lags if sensor_cfg.n_lags is not None else role_cfg.n_lags

    # Accept both int and string; treat 'auto' case-insensitively
    if isinstance(raw, str) and raw.strip().lower() == "auto":
        return max(1, int(train_rows) - int(n_forecasts) - 1)

    # Fall back to a normal integer
    return int(raw)


def build_future_from_train_index(train_df: pd.DataFrame,
                                  future_periods: int,
                                  freq: str,
                                  n_lags: int,
                                  n_forecasts: int,
                                  keep_history: bool = True) -> pd.DataFrame:
    """
    Create a future frame aligned to the resampled training ds grid.
    Includes a tail of history length (n_lags + n_forecasts) if keep_history=True.
    """
    ds = train_df["ds"]
    tz = getattr(ds.dt, "tz", None)
    assert len(ds) > 1, "Empty train ds"
    last_ds = ds.max()
    step = pd.tseries.frequencies.to_offset(freq)
    start_future = last_ds + step
    end_future = start_future + (future_periods - 1) * step if future_periods > 0 else last_ds
    fut_idx = pd.date_range(start=start_future, end=end_future, freq=freq, tz=tz)

    if keep_history:
        hist_len = max(0, int(n_lags) + int(n_forecasts))
        hist_tail = ds.iloc[-hist_len:] if hist_len > 0 else ds.iloc[[-1]]
        idx = pd.DatetimeIndex(hist_tail).append(pd.DatetimeIndex(fut_idx))
    else:
        idx = pd.DatetimeIndex(fut_idx)

    return pd.DataFrame({"ds": idx})

# --------------------------------------------------------------------------- #
# CovariateResolver
# --------------------------------------------------------------------------- #

class CovariateResolver:
    """
    Basic covariate resolution with alias & scale support.

    Example YAML:
      covariates:
        mixergy_charge:
          entity: sensor.current_charge
          scale: 0.01
        outdoor_temp_forecast: sensor.external_temperature
    """

    def __init__(self, iface: HAInterface, cov_map: Dict[str, Any]):
        self.iface = iface
        self.map = cov_map

    def _resolve(self, name: str) -> dict:
        val = self.map.get(name, name)
        if isinstance(val, str):
            return {"entity": val, "scale": 1.0}
        if isinstance(val, dict):
            return {
                "entity": val.get("entity", name),
                "scale": val.get("scale", 1.0),
                "attr": val.get("attr"),
                "forecast_attr": val.get("forecast_attr"),
                "units": val.get("units"),
            }
        return {"entity": str(val), "scale": 1.0}

    async def get_hist_series(self, cov_name: str, start: datetime, end: datetime, freq: str, how: str, sensor_cfg: Optional[SensorCfg] = None) -> pd.Series:
        meta = self._resolve(cov_name)
        entity_id = meta["entity"]
        logger.debug(
            "Covariate %s: fetching history for %s from %s to %s",
            cov_name,
            entity_id,
            start,
            end,
        )
        raw, _, _ = await self.iface.get_history(entity_id, start, end)
        df = normalise_history(raw)
        logger.debug(
            "Covariate %s: normalised history rows=%s", cov_name, len(df)
        )
        if df.empty:
            return pd.Series([], dtype=float)

        cov_series = df.set_index("ds")["value"]

        bin_cfg = is_binary_cov_name(cov_name, sensor_cfg) if sensor_cfg else False
        bin_auto = is_binary_like(cov_series)
        is_bin = bool(bin_cfg or bin_auto)

        cov_series = resample_covariate(cov_series, freq, is_binary=is_bin)

        logger.debug(
            "Covariate %s: binary=%s (cfg=%s auto=%s) rows=%d",
            cov_name,
            is_bin,
            bin_cfg,
            bin_auto,
            len(cov_series),
        )
        if is_bin:
            mn, mx = float(cov_series.min()), float(cov_series.max())
            uniq = sorted(pd.unique(cov_series.dropna()))[:4]
            logger.info(
                "Covariate %s treated as BINARY: min=%.1f max=%.1f uniques~%s",
                cov_name,
                mn,
                mx,
                uniq,
            )

        scale = float(meta.get("scale", 1.0))
        if not is_bin:
            cov_series = cov_series * scale
            logger.debug("Covariate %s: scaled by %s", cov_name, scale)

        out = cov_series
        logger.info(
            "Covariate %s: min=%s max=%s %s",
            cov_name,
            out.min(),
            out.max(),
            meta.get("units", ""),
        )
        logger.debug(
            "Covariate %s: obtained %s rows", cov_name, len(out)
        )
        logger.debug(
            "Covariate %s: head=%s", cov_name, out.head().to_dict()
        )
        return out

    async def get_future_series(self, cov_name: str, future_index: pd.DatetimeIndex, default: float = 0.0, sensor_cfg: Optional[SensorCfg] = None) -> pd.Series:
        meta = self._resolve(cov_name)
        entity_id = meta["entity"]
        scale = float(meta.get("scale", 1.0))
        attr_name = meta.get("forecast_attr")
    
        # If no forecast attribute is configured, fall back to current-state constant
        if not attr_name:
            val = await self.iface.get_state(entity_id)
            try:
                v = float(val)
            except (TypeError, ValueError):
                v = float(default)
            s = pd.Series(v, index=future_index)
            bin_cfg = is_binary_cov_name(cov_name, sensor_cfg) if sensor_cfg else False
            bin_auto = is_binary_like(s)
            is_bin = bool(bin_cfg or bin_auto)
            if is_bin:
                s = coerce_to_binary(s)
            else:
                s = s * scale
            logger.debug(
                "Covariate %s future: binary=%s (cfg=%s auto=%s) rows=%d",
                cov_name,
                is_bin,
                bin_cfg,
                bin_auto,
                len(s),
            )
            if is_bin:
                mn, mx = float(s.min()), float(s.max())
                uniq = sorted(pd.unique(s.dropna()))[:4]
                logger.info(
                    "Covariate %s treated as BINARY: min=%.1f max=%.1f uniques~%s",
                    cov_name,
                    mn,
                    mx,
                    uniq,
                )
            return s
    
        # 1) Fetch future series payload from HA
        payload = await self.iface.get_state(entity_id, attribute=attr_name) or []
    
        # 2) Normalise: handle both point forecasts and [start, end) intervals
        df = pd.DataFrame(payload)
    
        # Common field names seen in HA forecasts
        time_keys_point = [k for k in ["time", "datetime", "date", "at"] if k in df.columns]
        start_keys = [k for k in ["valid_from", "start", "from"] if k in df.columns]
        end_keys   = [k for k in ["valid_to", "end", "to", "until"] if k in df.columns]
    
        # Heuristic: pick the first numeric column as the value if not obviously named
        value_col_candidates = [c for c in df.columns if c.lower() in {"value", "price", "temperature", "temp", "y"}]
        if value_col_candidates:
            val_col = value_col_candidates[0]
        else:
            val_col = next((c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])), None)
    
        if val_col is None:
            # Nothing numeric -> fill defaults
            s = pd.Series(float(default), index=future_index)
            bin_cfg = is_binary_cov_name(cov_name, sensor_cfg) if sensor_cfg else False
            bin_auto = is_binary_like(s)
            is_bin = bool(bin_cfg or bin_auto)
            if is_bin:
                s = coerce_to_binary(s)
            else:
                s = s * scale
            logger.debug(
                "Covariate %s future: binary=%s (cfg=%s auto=%s) rows=%d",
                cov_name,
                is_bin,
                bin_cfg,
                bin_auto,
                len(s),
            )
            if is_bin:
                mn, mx = float(s.min()), float(s.max())
                uniq = sorted(pd.unique(s.dropna()))[:4]
                logger.info(
                    "Covariate %s treated as BINARY: min=%.1f max=%.1f uniques~%s",
                    cov_name,
                    mn,
                    mx,
                    uniq,
                )
            return s

        # Convert the model’s index to a DataFrame for joining
        target = pd.DataFrame({"ds": pd.to_datetime(future_index, utc=True)}).sort_values("ds")
    
        if time_keys_point:
            # --- Point forecasts: nearest-time match with a sensible tolerance ---
            dfp = df[[time_keys_point[0], val_col]].rename(columns={time_keys_point[0]: "ds", val_col: "value"})
            dfp["ds"] = pd.to_datetime(dfp["ds"], utc=True, errors="coerce")
            dfp = dfp.dropna(subset=["ds"]).sort_values("ds")
            # Tolerance: half the median step in target index
            if len(target) >= 2:
                step = (target["ds"].diff().dropna().median()) or pd.Timedelta("30min")
            else:
                step = pd.Timedelta("30min")
            out = pd.merge_asof(target, dfp, on="ds", direction="nearest", tolerance=step)["value"]
    
        elif start_keys and end_keys:
            # --- Interval forecasts: expand intervals to target slots they cover ---
            start_col, end_col = start_keys[0], end_keys[0]
            dfi = df[[start_col, end_col, val_col]].rename(
                columns={start_col: "start", end_col: "end", val_col: "value"}
            )
            dfi["start"] = pd.to_datetime(dfi["start"], utc=True, errors="coerce")
            dfi["end"]   = pd.to_datetime(dfi["end"],   utc=True, errors="coerce")
            dfi = dfi.dropna(subset=["start", "end"]).sort_values("start")
    
            # For each target ds, take the value from the interval where start <= ds < end
            # Vectorised: join-on-condition via merge_asof on start, then mask by end
            tmp = pd.merge_asof(target, dfi[["start", "end", "value"]], left_on="ds", right_on="start", direction="backward")
            mask = tmp["ds"].lt(tmp["end"])
            out = tmp["value"].where(mask)
    
        else:
            # Unknown shape: just fill default
            out = pd.Series(float(default), index=target.index)
    
        # 3) Scale / coerce binary, fill gaps deterministically, and return with the original index
        s = pd.to_numeric(out, errors="coerce")
        bin_cfg = is_binary_cov_name(cov_name, sensor_cfg) if sensor_cfg else False
        bin_auto = is_binary_like(s)
        is_bin = bool(bin_cfg or bin_auto)
        if is_bin:
            s = coerce_to_binary(s)
        else:
            s = s * scale
            s = s.ffill().fillna(float(default))
        logger.debug(
            "Covariate %s future: binary=%s (cfg=%s auto=%s) rows=%d",
            cov_name,
            is_bin,
            bin_cfg,
            bin_auto,
            len(s),
        )
        if is_bin:
            mn, mx = float(s.min()), float(s.max())
            uniq = sorted(pd.unique(s.dropna()))[:4]
            logger.info(
                "Covariate %s treated as BINARY: min=%.1f max=%.1f uniques~%s",
                cov_name,
                mn,
                mx,
                uniq,
            )
        s.index = future_index  # ensure identical index object
        return s

# --------------------------------------------------------------------------- #
# Model Backend (NeuralProphet)
# --------------------------------------------------------------------------- #

class NPBackend:
    def __init__(self,
                 n_lags: int,
                 n_forecasts: int,
                 seasonality_reg: float,
                 seasonality_mode: str = "additive",
                 learning_rate: Optional[float] = None,
                 country: Optional[str] = None):
        if NeuralProphet is None:
            raise RuntimeError(f"NeuralProphet import failed: {_NP_IMPORT_ERROR}")
        kw = dict(
            n_lags=n_lags,
            n_forecasts=n_forecasts,
            seasonality_mode=seasonality_mode,
            seasonality_reg=seasonality_reg,
        )
        if learning_rate is not None:
            kw["learning_rate"] = learning_rate
        self.model = NeuralProphet(**kw, drop_missing=True)
        if country:
            self.model.add_country_holidays(country)
        self.fitted = False

    def add_future_regressor(self, name: str, mode: str = "additive"):
        self.model.add_future_regressor(name, mode=mode)

    def add_lagged_regressor(self, name: str, n_lags: Optional[int] = None):
        self.model.add_lagged_regressor(name, n_lags=n_lags)

    def fit(self, df: pd.DataFrame, freq: str):
        self.model.fit(df, freq=freq, progress=None)
        self.fitted = True

    def make_future(self,
                    df: pd.DataFrame,
                    periods: int,
                    historic: bool = False,
                    future_regressors: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Wrapper around NP.make_future_dataframe that works across NP versions."""
        kw = dict(df=df, n_historic_predictions=historic, periods=periods)
        if future_regressors is not None:
            # NP ≥0.6 uses future_regressors=; some older builds used regressors_df=
            try:
                return self.model.make_future_dataframe(**kw, future_regressors=future_regressors)
            except TypeError:
                return self.model.make_future_dataframe(**kw, regressors_df=future_regressors)
        return self.model.make_future_dataframe(**kw)

    def predict(self, df_future: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(df_future)


# --------------------------------------------------------------------------- #
# Horizon helpers & publishing
# --------------------------------------------------------------------------- #

def horizon_steps(minutes_ahead: int, interval_min: int) -> int:
    return max(1, minutes_ahead // interval_min)


def horizon_agg(yhat_interval: Sequence[float], interval_min: int, minutes_ahead: int) -> float:
    steps = min(horizon_steps(minutes_ahead, interval_min), len(yhat_interval))
    return float(np.nansum(yhat_interval[:steps]))


def make_entity_name(prefix: str, base: str, suffix: Optional[str] = None) -> str:
    """Return a valid Home Assistant `entity_id for publishing state."""
    prefix = re.sub(r"^sensor[._]", "", prefix, flags=re.IGNORECASE)
    prefix = re.sub(r"[^a-z0-9_]+", "_", prefix.lower())
    base = re.sub(r"[^a-z0-9_]+", "_", base.lower())
    parts = [prefix + base]
    if suffix:
        parts.append(str(suffix))
    object_id = "_".join(parts)
    object_id = re.sub(r"_+", "_", object_id).strip("_")
        # Ensure PredAI prefix applied exactly once
    if not object_id.startswith("predai_"):
        object_id = "predai_" + object_id
    object_id = re.sub(r"^predai_predai_", "predai_", object_id)
    object_id = re.sub(r"_+", "_", object_id)
    return f"sensor.{object_id}"


def dict_from_series(index: Sequence[datetime], values: Sequence[float], tz: timezone) -> Dict[str, float]:
    return {
        ensure_utc(ts).astimezone(tz).strftime(TIME_FORMAT_HA): round(float(v), 3)
        for ts, v in zip(index, values)
    }


def daily_cumulative_series(index: Sequence[datetime], values: Sequence[float], tz: timezone) -> Dict[str, float]:
    cum = 0.0
    out: Dict[str, float] = {}
    current_day = None
    for ts, v in zip(index, values):
        lts = ensure_utc(ts).astimezone(tz)
        if current_day != lts.date():
            cum = 0.0
            current_day = lts.date()
        cum += max(float(v), 0.0)
        out[lts.strftime(TIME_FORMAT_HA)] = round(cum, 3)
    return out

def energy_already_used_today(df_cum: pd.DataFrame, tz: timezone) -> float:
    """Return the cumulative kWh that have been consumed *today*."""
    if df_cum.empty:
        return 0.0
    today = datetime.now(tz).date()
    today_rows = df_cum[df_cum["ds"].dt.date == today]
    if today_rows.empty:
        return 0.0
    return float(today_rows["value"].iloc[-1])


async def publish_forecasts(sensor: SensorCfg,
                            role_cfg: RoleCfg,
                            iface: HAInterface,
                            cfg: PredAIConfig,
                            ds_future: Sequence[datetime],
                            yhat_interval: Sequence[float],
                            yhat_level: Optional[Sequence[float]] = None,
                            metrics: Optional[dict] = None,
                            sensor_hist_cum: Optional[pd.DataFrame] = None):
    logger.info("Publishing forecasts for %s", sensor.name)
    tz = cfg.tz
    interval_min = sensor.interval or cfg.common_interval
    hist_df   = sensor_hist_cum if sensor_hist_cum is not None else pd.DataFrame()
    used_today = energy_already_used_today(hist_df, tz)
    prefix    = cfg.publish_prefix
    base_name = sensor.publish_name or sensor.name

    # ── Clip any forecast buckets that lie in the past ─────────────────────
    now_local = datetime.now(tz)
    for i, ts in enumerate(ds_future):
        if ensure_utc(ts).astimezone(tz) < now_local:
            yhat_interval[i] = 0.0
    # ───────────────────────────────────────────────────────────────────────

    yhat_interval = np.array(yhat_interval, dtype=float)
    yhat_interval = np.nan_to_num(yhat_interval, nan=0.0, posinf=0.0, neginf=0.0)
    yhat_interval = np.clip(yhat_interval, 0, None)  # no negatives

    publish_units = sensor.output_units or sensor.units
    if (sensor.units or "").lower() == "wh" and (publish_units or "").lower().endswith("kwh"):
        yhat_interval = yhat_interval / 1000.0

    if yhat_level is not None:
        yhat_level = np.array(yhat_level, dtype=float)
        yhat_level = np.nan_to_num(yhat_level, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    # *counter* sensors must continue from the last real reading
    # ------------------------------------------------------------------
    baseline = 0.0
    if sensor.source_is_cumulative:
        try:
            baseline = float(await iface.get_state(sensor.name, default=0.0))
        except Exception:
            baseline = 0.0

        # If the meter resets at midnight, start the forecast curve at the
        # energy already used today so orange & blue meet at "now".
        if sensor.reset_daily:
            baseline = used_today

    cum_from_now = baseline + np.cumsum(yhat_interval)
    if sensor.reset_daily and sensor.source_is_cumulative:
        midnight = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        for i, ts in enumerate(ds_future):
            if ensure_utc(ts).astimezone(tz) >= midnight:
                reset_val = cum_from_now[i-1] if i > 0 else baseline
                cum_from_now[i:] -= reset_val
                break
    # - Daily cumulative forecast starting from the energy already used today so
    #   the curve meets the live meter reading at 'now'.
    # Re-build daily cumulative after clipping & baseline offset
    daily_cum = daily_cumulative_series(ds_future, yhat_interval, tz)
    today_str = now_local.strftime("%Y-%m-%d")
    for ts_iso in list(daily_cum.keys()):
        if ts_iso.startswith(today_str):
            daily_cum[ts_iso] += used_today

    ser_interval = dict_from_series(ds_future, yhat_interval, tz)
    ser_cum = dict_from_series(ds_future, cum_from_now, tz)

    model_ts_iso = datetime.now(timezone.utc).astimezone(tz).isoformat()
    meta = {
        "model_ts": model_ts_iso,
        "model_backend": role_cfg.model_backend,
        "training_rows": metrics.get("training_rows") if metrics else None,
        "mae_recent": metrics.get("mae_recent") if metrics else None,
    }

    state_class = ("total_increasing" if sensor.source_is_cumulative else role_cfg.publish_state_class)

    if sensor.publish_interval:
        ent_interval = make_entity_name(prefix, base_name, "interval")
        await iface.set_state(
            ent_interval,
            state=round(float(yhat_interval[0]) if len(yhat_interval) else 0.0, 3),
            attributes={
                "unit_of_measurement": publish_units,
                "state_class": "measurement",
                "forecast_series": ser_interval,
                **meta,
            },
        )

    if sensor.publish_cumulative:
        ent_cum = make_entity_name(prefix, base_name, "cum")
        await iface.set_state(
            ent_cum,
            state=round(float(cum_from_now[-1]) if len(cum_from_now) else 0.0, 3),
            attributes={
                "unit_of_measurement": publish_units,
                "state_class": state_class,
                "forecast_series": ser_cum,
                **meta,
            },
        )

    if sensor.publish_daily_cumulative:
        ent_daily = make_entity_name(prefix, base_name, "daily_cum")
        today_str = datetime.now(tz).strftime("%Y-%m-%d")
        todays = {k: v for k, v in daily_cum.items() if k.startswith(today_str)}
        state_val = list(todays.values())[-1] if todays else list(daily_cum.values())[-1]
        await iface.set_state(
            ent_daily,
            state=round(float(state_val), 3),
            attributes={
                "unit_of_measurement": publish_units,
                "state_class": state_class,
                "forecast_series": daily_cum,
                **meta,
            },
        )

        # --------------------------------------------------------------
        #  Preserve the previous forecast curve → “…_curve_yesterday”
        # --------------------------------------------------------------
        ent_curve      = make_entity_name(prefix, base_name, "pred_curve")
        ent_curve_prev = make_entity_name(prefix, base_name, "curve_yesterday")

        old_item = await iface.api_call("GET", f"/api/states/{ent_curve}")
        if old_item:
            await iface.set_state(
                ent_curve_prev,
                state=old_item.get("state", 0),
                attributes=old_item.get("attributes", {}),
            )

        await iface.set_state(
            ent_curve,
            state=round(list(daily_cum.values())[-1], 3),
            attributes={
                "unit_of_measurement": publish_units,
                "state_class": state_class,
                "forecast_series": daily_cum,
                **meta,
            },
        )

        # --------------------------------------------------------------
        #  One-shot "initial" curve -> "…_curve_initial"
        # --------------------------------------------------------------
        def _same_day(ts1: str, ts2: str) -> bool:
            try:
                d1 = timestr_to_datetime(ts1).astimezone(tz).date()
                d2 = timestr_to_datetime(ts2).astimezone(tz).date()
                return d1 == d2
            except Exception:
                return False

        ent_curve_init = make_entity_name(prefix, base_name, "pred_curve_initial")
        existing = await iface.api_call("GET", f"/api/states/{ent_curve_init}")
        need_update = True
        if existing and "attributes" in existing:
            old_ts = existing["attributes"].get("model_ts")
            need_update = not _same_day(old_ts, model_ts_iso)

        if need_update:
            await iface.set_state(
                ent_curve_init,
                state=round(list(daily_cum.values())[-1], 3),
                attributes={
                    "unit_of_measurement": publish_units,
                    "state_class": state_class,
                    "forecast_series": daily_cum,
                    "generated_from": make_entity_name(prefix, base_name, "interval"),
                    **meta,
                },
            )

        # --------------------------------------------------------------
        #  Rolling 7‑day buffer of initial curves  → “…_pred_curve_initial_7d”
        # --------------------------------------------------------------
        ent_curve_hist = make_entity_name(prefix, base_name, "pred_curve_initial_7d")
        today_key = now_local.strftime("%Y-%m-%d")

        # 1.  Load existing map (if any)
        prev_item = await iface.api_call("GET", f"/api/states/{ent_curve_hist}")
        hist_map: Dict[str, Any] = {}
        if prev_item and "attributes" in prev_item:
            hist_map = prev_item["attributes"].get("forecast_series_map", {}) or {}

        # 2.  Prune to the most‑recent six previous days
        cutoff = now_local.date() - timedelta(days=6)
        hist_map = {
            d: s for d, s in hist_map.items()
            if datetime.fromisoformat(d).date() >= cutoff
        }

        # 3.  Add today’s curve only if not already present
        if today_key not in hist_map:
            hist_map[today_key] = daily_cum

        # 4.  Publish / overwrite the rolling sensor
        await iface.set_state(
            ent_curve_hist,
            state=round(list(daily_cum.values())[-1], 3),
            attributes={
                "unit_of_measurement": publish_units,
                "state_class": state_class,
                "forecast_series_map": hist_map,
                **meta,
            },
        )

    # Horizon scalars
    for m in cfg.horizons:
        suffix = f"pred_{m//60}h"
        ent_h = make_entity_name(prefix, base_name, suffix)
        if sensor.train_target == "level":  # e.g., temperature
            arr = np.array(yhat_level if yhat_level is not None else yhat_interval)
            steps = min(horizon_steps(m, interval_min), len(arr))
            val = arr[steps - 1]
        else:
            val = horizon_agg(yhat_interval, interval_min, m)
        await iface.set_state(
            ent_h,
            state=round(float(val), 3),
            attributes={
                "unit_of_measurement": publish_units,
                "state_class": "measurement",
                "generated_from": make_entity_name(prefix, base_name, "interval"),
                **meta,
            },
        )

    logger.info("Finished publishing forecasts for %s", sensor.name)


# --------------------------------------------------------------------------- #
# Sensor job execution
# --------------------------------------------------------------------------- #

async def run_sensor_job(sensor: SensorCfg,
                         role_cfg: RoleCfg,
                         cfg: PredAIConfig,
                         iface: HAInterface,
                         cov_res: CovariateResolver,
                         db: Optional[HistoryDB]) -> None:
    interval_min = sensor.interval or cfg.common_interval
    freq = f"{interval_min}min"
    tz = cfg.tz

    # Round now to interval boundary
    now = datetime.now(timezone.utc).astimezone(tz).replace(second=0, microsecond=0)
    minute_floor = (now.minute // interval_min) * interval_min
    now = now.replace(minute=minute_floor)

    start_hist = now - timedelta(days=sensor.days_hist)
    end_hist = now

    logger.info("Sensor %s: fetching history %s → %s", sensor.name, start_hist, end_hist)
    raw_hist, st, en = await iface.get_history(sensor.name, start_hist, end_hist)
    df = normalise_history(raw_hist)
    logger.info(
        "Sensor %s: history rows=%s first=%s last=%s",
        sensor.name,
        len(df),
        df["ds"].min() if not df.empty else None,
        df["ds"].max() if not df.empty else None,
    )
    if not df.empty:
        logger.info(
            "Sensor %s: min=%s max=%s %s",
            sensor.name,
            df["value"].min(),
            df["value"].max(),
            sensor.units or "",
        )

    # DB merge
    if sensor.database and db:
        tname = sensor.name.replace(".", "_")
        prev = db.get_history(tname)
        if not df.empty:
            tmp = df.rename(columns={"value": "y"})[["ds", "y"]]
            prev = db.store_history(tname, tmp, prev)
        oldest = now - timedelta(days=sensor.max_age)
        db.cleanup_table(tname, oldest)
        if not prev.empty:
            prev = prev.sort_values("ds")
            prev = prev.rename(columns={"y": "value"})
            df = prev
        logger.info(
            "Sensor %s: dataset after DB merge rows=%s", sensor.name, len(df)
        )

    if df.empty:
        logger.warning("Sensor %s: no data; skipping.", sensor.name)
        return

    df_cum_raw = df.copy()

    agg = sensor.effective_aggregation(role_cfg)      # keep "sum" for energy

    # --------------------------------------------------
    # 1.  Convert cumulative counter → interval (daily-aware)
    # --------------------------------------------------
    if sensor.source_is_cumulative:
        raw_df = df.rename(columns={"ds": "ts", "value": "val"})[["ts", "val"]]
        logging.debug("Cumulative->interval(gap-aware) on %d rows", len(raw_df))
        _entity_id = getattr(sensor, "entity_id", getattr(sensor, "name", "unknown"))
        _unit = getattr(sensor, "units", None)
        _state_class = getattr(sensor, "state_class", None)

        y30 = cumulative_to_interval_gapaware_with_daily(
            raw_df=raw_df,
            entity_id=_entity_id,
            unit=_unit,
            state_class=_state_class,
            cfg=cfg,
            resample_freq=freq,
        )
        logging.info(
            "Sensor %s: after daily-aware conversion %d rows from %s",
            _entity_id,
            len(y30),
            freq,
        )

        # --- Outlier capping on interval y (safe, positives-only) ---
        outlier_q = getattr(sensor, "outlier_cap_q", 0.995)

        if outlier_q is not None:
            # Compute cap from strictly-positive values only
            y_series = y30["y"].astype("float32")
            pos = y_series[y_series > 0]
            nonzero_share = 100.0 * (len(pos) / max(1, len(y_series)))
            cap_val = float(pos.quantile(outlier_q)) if len(pos) > 0 else float("nan")

            if np.isfinite(cap_val) and cap_val > 0.0:
                y_series = y_series.clip(lower=0.0, upper=cap_val)
                logger.info(
                    "Outlier cap applied at p%.2f of positives=%.3f; y.max() post-cap=%.3f; nonzero_share=%.2f%%",
                    outlier_q * 100.0,
                    cap_val,
                    float(y_series.max()),
                    nonzero_share,
                )
                y30["y"] = y_series
            else:
                logger.info(
                    "Outlier capping skipped (cap<=0 or insufficient positives). nonzero_share=%.2f%%",
                    nonzero_share,
                )

        # Guard: if preprocessing flattened the target, stop early to avoid NP crash.
        if y30["y"].nunique(dropna=True) < 2:
            msg = "Target became constant after preprocessing (likely extreme sparsity). Skipping training."
            logger.error(msg)
            return

        df = y30
    else:
        # --------------------------------------------------
        # 2.  Resample the interval series (sum / mean / last)
        # --------------------------------------------------
        df = resample_sensor(df, freq, agg)
        logger.info(
            "Sensor %s: after resample %s rows from %s", sensor.name, len(df), freq
        )

        # --------------------------------------------------
        # 3.  Rename value → y (now exists for *all* sensors)
        # --------------------------------------------------
        df = df.rename(columns={"value": "y"})

    if sensor.subtract:
        for sub_name in sensor.subtract:
            logger.info("Sensor %s: subtracting %s", sensor.name, sub_name)
            raw_sub, _, _ = await iface.get_history(sub_name, start_hist, end_hist)
            sub_df = normalise_history(raw_sub)
            if sensor.source_is_cumulative or sensor.incrementing:
                sub_df = cumulative_to_interval(sub_df, sensor.reset_detection, sensor.max_increment)
                sub_df["value"] = sub_df["y"]
            sub_df = resample_sensor(sub_df, freq, agg)
            sub_df = sub_df.rename(columns={"value": "y"})
            sub_df["y"] = pd.to_numeric(sub_df["y"], errors="coerce").fillna(0.0)
            df = subtract_set(df, sub_df, inc=sensor.incrementing or sensor.source_is_cumulative)

    # Power->energy (heuristic)
    if (not sensor.source_is_cumulative) and sensor.train_target == "interval":
        units_lower = (sensor.units or "").lower()
        if "w" in units_lower:  # W or kW
            if df["y"].max() > 50:  # assume W
                df["y"] = df["y"] / 1000.0
            df["y"] = df["y"] * (interval_min / 60.0)  # kWh per bucket
            if not sensor.output_units:
                sensor.output_units = "kWh"

    # Clean
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    logger.debug(
        "Sensor %s: cleaned data rows=%s", sensor.name, len(df)
    )

    log_applied = False
    if sensor.log_transform:
        df = apply_log_transform(df)
        log_applied = True
        logger.debug("Sensor %s: applied log transform", sensor.name)

    # Training frame
    train_df = df[["ds", "y"]].copy()
    logger.info(
        "Sensor %s: training frame %s rows", sensor.name, len(train_df)
    )
    summarise_df(f"train.{sensor.name}", train_df)
    if not train_df.empty:
        logger.info(
            "Sensor %s: training min=%s max=%s %s",
            sensor.name,
            train_df["y"].min(),
            train_df["y"].max(),
            sensor.output_units or sensor.units or "",
        )

    if role_cfg.model_backend == "neuralprophet":
        steps = sensor.future_periods if sensor.future_periods is not None else max(
            horizon_steps(m, interval_min) for m in cfg.horizons
        )
        # Resolve n_lags (supports 'auto')
        n_lags_eff = resolve_n_lags(sensor, role_cfg, len(train_df), steps)
        logger.info("Sensor %s: using n_lags=%s (mode=%s)",
                    sensor.name, n_lags_eff,
                    "auto" if (isinstance(sensor.n_lags or role_cfg.n_lags, str)
                               and str(sensor.n_lags or role_cfg.n_lags).lower()=="auto") else "fixed")

        # Now that we know the effective n_lags, do a proper history sufficiency check
        min_needed = n_lags_eff + steps + 1
        if len(train_df) < min_needed:
            logger.warning(
                "Sensor %s: insufficient history for n_lags=%s & steps=%s "
                "(have %s rows, need ≥ %s); skipping model.",
                sensor.name, n_lags_eff, steps, len(train_df), min_needed
            )
            return

        backend = NPBackend(
            n_lags=n_lags_eff,
            n_forecasts=steps,
            seasonality_reg=sensor.effective_seasonality_reg(role_cfg),
            seasonality_mode=sensor.effective_seasonality_mode(role_cfg),
            learning_rate=sensor.effective_learning_rate(role_cfg),
            country=sensor.country,
        )

        # lagged covariates (including both)
        for cov in list(dict.fromkeys(list(sensor.covariates_lagged) + list(sensor.covariates_both))):
            s = await cov_res.get_hist_series(cov, start_hist, end_hist, freq, agg, sensor_cfg=sensor)
            if not s.empty:
                train_df = train_df.merge(s.rename(cov), left_on="ds", right_index=True, how="left")
                logger.debug(
                    "Covariate %s lagged: merged %s rows", cov, len(s)
                )
                is_binary = bool(is_binary_cov_name(cov, sensor) or is_binary_like(s))
                n_lags_for_cov = _lags_for_covariate(cov, n_lags_eff, sensor)
                backend.add_lagged_regressor(cov, n_lags=n_lags_for_cov)
                logger.info(
                    "Added lagged regressor %s with n_lags=%d (binary=%s)",
                    cov,
                    n_lags_for_cov,
                    is_binary,
                )
            else:
                logger.debug("Covariate %s lagged: no history.", cov)

        # future covariates (including both)
        for cov in list(dict.fromkeys(list(sensor.covariates_future) + list(sensor.covariates_both))):
            s = await cov_res.get_hist_series(cov, start_hist, end_hist, freq, agg, sensor_cfg=sensor)
            if not s.empty:
                train_df = train_df.merge(s.rename(cov), left_on="ds", right_index=True, how="left")
                logger.debug(
                    "Covariate %s future: merged %s rows", cov, len(s)
                )
            else:
                train_df[cov] = np.nan
                logger.debug("Covariate %s future: no history, filled NaN", cov)
            backend.add_future_regressor(cov, mode="additive")

        # ensure regressor columns exist even if HA returned no history
        cov_cols = list(dict.fromkeys(list(sensor.covariates_lagged) + list(sensor.covariates_future) + list(sensor.covariates_both)))
        for cov in cov_cols:
            if cov not in train_df.columns:
                train_df[cov] = 0.0
        if cov_cols:
            # Forward-fill then back-fill
            train_df[cov_cols] = train_df[cov_cols].ffill().bfill()
            # If any NAs remain (e.g., covariate starts late), fill with column medians
            if train_df[cov_cols].isna().any().any():
                med = train_df[cov_cols].median(numeric_only=True)
                train_df[cov_cols] = train_df[cov_cols].fillna(med)
            # Final safety net
            train_df[cov_cols] = train_df[cov_cols].fillna(0.0)

        train_df = train_df.dropna(subset=["y"])
        _delta = train_df["ds"].diff().dropna().value_counts()
        assert not _delta.empty and _delta.index[0] == pd.Timedelta(minutes=30), f"Cadence not 30min: {dict(_delta)}"

        # Log a summary after imputation
        summarise_df(f"train_imputed.{sensor.name}", train_df)

        _npos = int((train_df["y"] > 0).sum()) if len(train_df) else 0
        logging.info(
            "Train y: rows=%d, positives=%d (%.2f%%), y.max=%.3f, y.sum=%.3f",
            len(train_df),
            _npos,
            (100.0 * _npos / max(1, len(train_df))),
            float(0 if train_df.empty else train_df["y"].max()),
            float(0 if train_df.empty else train_df["y"].sum()),
        )

        # Fit
        backend.fit(train_df, freq=freq)
        logger.info("Sensor %s: model trained", sensor.name)

        # ------------------------------------------------------------------
        # Future frame + prediction — pass exogenous columns + future values
        # ------------------------------------------------------------------
        steps = sensor.future_periods if sensor.future_periods is not None else max(
            horizon_steps(m, interval_min) for m in cfg.horizons
        )
        last_ts = train_df["ds"].max()

        df_future = build_future_from_train_index(
            train_df=train_df,
            future_periods=steps,
            freq=freq,
            n_lags=n_lags_eff,
            n_forecasts=steps,
            keep_history=True,
        )
        df_future = df_future.merge(train_df[["ds", "y"] + cov_cols], on="ds", how="left")
        assert df_future["ds"].is_monotonic_increasing
        fut_idx = pd.DatetimeIndex(df_future.loc[df_future["ds"] > last_ts, "ds"])
        fut_names = [c for c in (list(sensor.covariates_future) + list(sensor.covariates_both)) if c in cov_cols]
        for cov in fut_names:
            s = await cov_res.get_future_series(cov, fut_idx, default=0.0, sensor_cfg=sensor)
            bin_cfg = is_binary_cov_name(cov, sensor)
            bin_auto = is_binary_like(s)
            is_bin = bool(bin_cfg or bin_auto)
            s = resample_covariate(s, freq, is_binary=is_bin)
            logger.debug(
                "Covariate %s: binary=%s (cfg=%s auto=%s) rows=%d",
                cov,
                is_bin,
                bin_cfg,
                bin_auto,
                len(s),
            )
            if is_bin:
                mn, mx = float(s.min()), float(s.max())
                uniq = sorted(pd.unique(s.dropna()))[:4]
                logger.info(
                    "Covariate %s treated as BINARY: min=%.1f max=%.1f uniques~%s",
                    cov,
                    mn,
                    mx,
                    uniq,
                )
            cov_df = s.rename(cov).reset_index().rename(columns={"index": "ds"})
            df_future = df_future.merge(cov_df, on="ds", how="left")
        for name in cov_cols:
            if name in df_future:
                df_future[name] = df_future[name].ffill().bfill(limit=1)
        _expected = len(df_future)
        for col in (["y"] if "y" in df_future.columns else []) + list(cov_cols):
            if col in df_future:
                assert len(df_future[col]) == _expected, f"Future column length mismatch: {col} {_expected=} got {len(df_future[col])}"
        _delta = df_future["ds"].diff().dropna().value_counts()
        assert not _delta.empty and _delta.index[0] == pd.Timedelta(minutes=30), f"Future cadence not 30min: {dict(_delta)}"
        deltas = df_future["ds"].diff().dropna().value_counts()
        logger.info("DF future.%s: rows=%d cadence_ok=%s first3=%s last3=%s",
                    sensor.name, len(df_future),
                    (not deltas.empty and deltas.index[0] == pd.Timedelta(minutes=30)),
                    list(df_future['ds'].head(3)), list(df_future['ds'].tail(3)))
        fcst = backend.predict(df_future)
        fcst["ds"] = pd.to_datetime(fcst["ds"], utc=True)
        summarise_df(f"forecast.{sensor.name}", fcst)

        # Find the first row whose yhat1..N are all non-NaN
        yhat_cols = sorted([c for c in fcst.columns if c.startswith("yhat")], key=lambda s: int(s[4:]))
        if not yhat_cols:
            raise RuntimeError("No yhat columns present in forecast output")
        mask_complete = fcst[yhat_cols].notna().all(axis=1)
        if mask_complete.any():
            first_future = fcst.loc[mask_complete].iloc[0]
        else:
            # fall back to last row; some NP versions only fill at the end
            first_future = fcst.iloc[-1]
        yhat_int = first_future[yhat_cols].to_numpy()
        if log_applied:
            yhat_int = invert_log_transform(yhat_int, True)

        # Future timestamps are exactly the fut_idx constructed above
        ds_future = [ts.to_pydatetime() for ts in fut_idx]

        metrics = {"training_rows": int(len(train_df)), "mae_recent": None}
        await publish_forecasts(sensor, role_cfg, iface, cfg, ds_future, yhat_int,
                                metrics=metrics, sensor_hist_cum=df_cum_raw)
        logger.info("Sensor %s: forecasting complete", sensor.name)

    else:
        logger.error("Unsupported backend %s for sensor %s", role_cfg.model_backend, sensor.name)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

async def predai_main():
    cfg = load_config(DEFAULT_CONFIG_PATH)

    # read raw for HA creds & initial cov_map
    try:
        with open(DEFAULT_CONFIG_PATH, "r") as f:
            raw = yaml.safe_load(f) or {}
    except Exception:
        raw = {}
    ha_url = raw.get("ha_url")
    ha_key = raw.get("ha_key") or os.environ.get("SUPERVISOR_TOKEN")

    iface = HAInterface(ha_url, ha_key)
    cov_res = CovariateResolver(iface, cfg.cov_map)
    db = HistoryDB(DEFAULT_DB_PATH)

    try:
        while True:
            logger.info("PredAI cycle start.")
            # hot-reload config
            cfg = load_config(DEFAULT_CONFIG_PATH)
            # refresh covariate map
            cov_res.map = cfg.cov_map

            for s in cfg.sensors:
                role_cfg = cfg.roles.get(s.role, RoleCfg())
                try:
                    await run_sensor_job(s, role_cfg, cfg, iface, cov_res, db)
                except Exception as e:
                    logger.exception("Sensor job failed for %s: %s", s.name, e)

            now_str = datetime.now(timezone.utc).isoformat()
            await iface.set_state(
                "sensor.predai_last_run",
                state=now_str,
                attributes={"unit_of_measurement": "time"},
            )

            logger.info("PredAI sleeping %s minutes.", cfg.update_every)
            # Sleep minute chunks; break early if heartbeat lost. Output a
            # progress log each minute so hosting platforms don't kill the
            # process for lack of output. Log once before the first sleep so
            # there is never a full minute with no output.
            logger.info(
                "PredAI sleep progress: 0/%s minutes",
                cfg.update_every,
            )
            for i in range(cfg.update_every):
                await asyncio.sleep(60)
                last_run = await iface.get_state("sensor.predai_last_run")
                if last_run is None:
                    logger.warning("PredAI heartbeat lost; restarting early.")
                    break
                logger.info(
                    "PredAI sleep progress: %s/%s minutes",
                    i + 1,
                    cfg.update_every,
                )

    finally:
        await iface.close()
        db.close()


def main():
    asyncio.run(predai_main())


if __name__ == "__main__":
    main()
