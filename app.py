import math
import time
import sqlite3
import json
import random
from datetime import datetime, date

import streamlit as st
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# CONFIG
# ============================================================
APP_TITLE = "JAMS Capital Options Terminal"
DB_FILE = "jams_options_snapshots.sqlite"
CACHE_DB = "jams_yf_cache.sqlite"

CONTRACT_MULTIPLIER = 100

SHORT_DAYS = 14
MID_DAYS = 180
LONG_DAYS = 365

# Cloud-safe defaults
SPOT_TTL_SEC_DEFAULT = 10 * 60          # 10 min for spot
EXPIRIES_TTL_SEC_DEFAULT = 6 * 60 * 60  # 6 hours
CHAIN_TTL_SEC_DEFAULT = 6 * 60 * 60     # 6 hours

MIN_REFRESH_INTERVAL_SEC = 30
MAX_RETRIES = 3

st.set_page_config(layout="wide", page_title=APP_TITLE, page_icon="📈")


# ============================================================
# BLOOMBERG BLACK THEME (BaseWeb-correct)
# ============================================================
CSS = r"""
<style>
html, body, .stApp,
[data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main,
section.main, .block-container,
header[data-testid="stHeader"], [data-testid="stToolbar"], div[data-testid="stDecoration"]{
  background: #000000 !important;
  color: #E6E6E6 !important;
}
section[data-testid="stSidebar"]{
  background:#000000 !important;
  border-right:1px solid rgba(255,153,28,0.35) !important;
}
section[data-testid="stSidebar"] *{ color:#E6E6E6 !important; }

h1,h2,h3,h4,h5,h6{ color:#FF991C !important; font-weight: 950 !important; }

div[data-baseweb="input"]{
  background:#0B0B0B !important;
  border:1px solid rgba(255,153,28,0.45) !important;
  border-radius:10px !important;
}
div[data-baseweb="input"] input{
  background:#0B0B0B !important;
  color:#E6E6E6 !important;
  -webkit-text-fill-color:#E6E6E6 !important;
  caret-color:#FF991C !important;
}
div[data-baseweb="input"] input::placeholder{
  color:rgba(230,230,230,0.55) !important;
  -webkit-text-fill-color:rgba(230,230,230,0.55) !important;
}

div[data-testid="stNumberInput"] button{
  background:#0B0B0B !important;
  border:1px solid rgba(255,153,28,0.35) !important;
}
div[data-testid="stNumberInput"] button svg{ fill:#FF991C !important; }

div[data-baseweb="textarea"]{
  background:#0B0B0B !important;
  border:1px solid rgba(255,153,28,0.45) !important;
  border-radius:10px !important;
}
div[data-baseweb="textarea"] textarea{
  background:#0B0B0B !important;
  color:#E6E6E6 !important;
  -webkit-text-fill-color:#E6E6E6 !important;
  caret-color:#FF991C !important;
}
div[data-baseweb="textarea"] textarea::placeholder{
  color:rgba(230,230,230,0.55) !important;
  -webkit-text-fill-color:rgba(230,230,230,0.55) !important;
}

div[data-baseweb="select"] > div{
  background:#0B0B0B !important;
  border:1px solid rgba(255,153,28,0.45) !important;
  border-radius:10px !important;
}
div[data-baseweb="select"] *{
  color:#E6E6E6 !important;
  -webkit-text-fill-color:#E6E6E6 !important;
}
div[data-baseweb="select"] svg{ fill:#FF991C !important; }

div[role="listbox"]{
  background:#000000 !important;
  border:1px solid rgba(255,153,28,0.65) !important;
  border-radius:10px !important;
}
li[role="option"]{ background:#000000 !important; color:#E6E6E6 !important; }
li[role="option"]:hover{ background:#121212 !important; }
li[role="option"][aria-selected="true"]{
  background:rgba(255,153,28,0.18) !important;
  color:#FF991C !important;
}

div[data-testid="stSlider"] [role="slider"]{ background:#FF991C !important; }

div.stButton > button{
  background:#FF991C !important;
  color:#000000 !important;
  font-weight: 950 !important;
  border:0 !important;
  border-radius:10px !important;
  padding:0.55rem 0.9rem !important;
}
button[data-baseweb="tab"]{ background:#000000 !important; color:#E6E6E6 !important; border-bottom:2px solid transparent !important; }
button[data-baseweb="tab"][aria-selected="true"]{ color:#FF991C !important; border-bottom:2px solid #FF991C !important; }

div[data-testid="stMetric"]{
  background:#0B0B0B !important;
  border:1px solid rgba(255,153,28,0.35) !important;
  border-radius:12px !important;
  padding:14px !important;
}
div[data-testid="stMetricLabel"]{ color:#B8B8B8 !important; }
div[data-testid="stMetricValue"]{ color:#E6E6E6 !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    f"""
    <div style="padding:16px; background:#000000; border:1px solid rgba(255,153,28,0.55);
                border-radius:12px; text-align:center;">
      <div style="font-size:28px; font-weight:950; color:#FF991C;">{APP_TITLE}</div>
      <div style="margin-top:6px; font-weight:800; color:#00ff41;">REAL DATA ONLY (Yahoo via yfinance) — CLOUD SAFE MODE</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# Persistent cache (sqlite)
# ============================================================
def cache_init():
    con = sqlite3.connect(CACHE_DB)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS kv_cache (
            k TEXT PRIMARY KEY,
            ts_utc INTEGER,
            payload_json TEXT
        )
    """)
    con.commit()
    con.close()

def cache_get(key: str, ttl_sec: int, allow_stale=False):
    cache_init()
    con = sqlite3.connect(CACHE_DB)
    cur = con.cursor()
    cur.execute("SELECT ts_utc, payload_json FROM kv_cache WHERE k=?", (key,))
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    ts_utc, payload = row
    age = int(time.time()) - int(ts_utc)
    if (not allow_stale) and age > int(ttl_sec):
        return None
    try:
        return json.loads(payload)
    except Exception:
        return None

def cache_set(key: str, obj: dict):
    cache_init()
    con = sqlite3.connect(CACHE_DB)
    cur = con.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO kv_cache (k, ts_utc, payload_json) VALUES (?,?,?)",
        (key, int(time.time()), json.dumps(obj))
    )
    con.commit()
    con.close()

def cache_key(*parts):
    return "::".join([str(p) for p in parts])

def _sleep_backoff(attempt: int):
    time.sleep(min(8.0, (1.4 ** attempt) + random.random()))

# ============================================================
# yfinance safe wrappers
# ============================================================
def yf_history_cached(ticker: str, period: str, interval: str, ttl_sec: int):
    key = cache_key("history", ticker, period, interval)
    cached = cache_get(key, ttl_sec=ttl_sec)
    if cached:
        return pd.DataFrame(cached["rows"])

    last_err = None
    for i in range(MAX_RETRIES):
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period=period, interval=interval)
            if hist is None or hist.empty:
                raise RuntimeError("No price data returned.")
            df = hist.reset_index()
            cache_set(key, {"rows": df.to_dict(orient="records")})
            return df
        except YFRateLimitError as e:
            last_err = e
            _sleep_backoff(i + 1)
        except Exception as e:
            last_err = e
            _sleep_backoff(i + 1)

    stale = cache_get(key, ttl_sec=ttl_sec, allow_stale=True)
    if stale:
        return pd.DataFrame(stale["rows"])
    raise last_err if last_err else RuntimeError("Failed to fetch history.")

def yf_expiries_cached(ticker: str, ttl_sec: int):
    key = cache_key("expiries", ticker)
    cached = cache_get(key, ttl_sec=ttl_sec)
    if cached:
        return cached["expiries"]

    last_err = None
    for i in range(MAX_RETRIES):
        try:
            t = yf.Ticker(ticker)
            exps = t.options
            if not exps:
                raise RuntimeError("No expiries returned.")
