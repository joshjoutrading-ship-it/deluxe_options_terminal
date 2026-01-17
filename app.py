import math
import time
import sqlite3
import json
import random
from datetime import datetime, date, timedelta
from pathlib import Path

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
CACHE_DB = "jams_yf_cache.sqlite"      # persistent cache to survive reruns
CONTRACT_MULTIPLIER = 100

SHORT_DAYS = 14
MID_DAYS = 180
LONG_DAYS = 365

# IMPORTANT: throttle to respect free sources
DEFAULT_CACHE_TTL_SEC = 15 * 60        # 15 minutes persistent cache
MIN_REFRESH_INTERVAL_SEC = 30          # do not allow spam refresh
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

/* BaseWeb Inputs */
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

/* NumberInput buttons */
div[data-testid="stNumberInput"] button{
  background:#0B0B0B !important;
  border:1px solid rgba(255,153,28,0.35) !important;
}
div[data-testid="stNumberInput"] button svg{ fill:#FF991C !important; }

/* Textarea */
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

/* Select */
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

/* Dropdown popover */
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

/* Slider */
div[data-testid="stSlider"] [role="slider"]{ background:#FF991C !important; }

/* Buttons */
div.stButton > button{
  background:#FF991C !important;
  color:#000000 !important;
  font-weight: 950 !important;
  border:0 !important;
  border-radius:10px !important;
  padding:0.55rem 0.9rem !important;
}
div.stButton > button:hover{ filter:brightness(0.95); }

/* Tabs */
button[data-baseweb="tab"]{ background:#000000 !important; color:#E6E6E6 !important; border-bottom:2px solid transparent !important; }
button[data-baseweb="tab"][aria-selected="true"]{ color:#FF991C !important; border-bottom:2px solid #FF991C !important; }

/* Metrics & tables */
div[data-testid="stMetric"]{
  background:#0B0B0B !important;
  border:1px solid rgba(255,153,28,0.35) !important;
  border-radius:12px !important;
  padding:14px !important;
}
div[data-testid="stMetricLabel"]{ color:#B8B8B8 !important; }
div[data-testid="stMetricValue"]{ color:#E6E6E6 !important; }

div[data-testid="stDataFrame"]{
  background:#000000 !important;
  border:1px solid rgba(255,153,28,0.35) !important;
  border-radius:12px !important;
  overflow:hidden !important;
}
div[data-testid="stDataFrame"] thead tr th{ background:#0B0B0B !important; color:#E6E6E6 !important; }
div[data-testid="stDataFrame"] tbody tr td{ background:#000000 !important; color:#E6E6E6 !important; }

hr{ border:none !important; border-top:1px solid rgba(255,153,28,0.25) !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    f"""
    <div style="padding:16px; background:#000000; border:1px solid rgba(255,153,28,0.55);
                border-radius:12px; text-align:center;">
      <div style="font-size:28px; font-weight:950; color:#FF991C;">{APP_TITLE}</div>
      <div style="margin-top:6px; font-weight:800; color:#00ff41;">REAL CHAIN DATA ONLY (Yahoo Finance via yfinance) — RATE-LIMIT SAFE MODE</div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Persistent cache (sqlite) to reduce yfinance hits
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

def cache_get(key: str, ttl_sec: int):
    cache_init()
    con = sqlite3.connect(CACHE_DB)
    cur = con.cursor()
    cur.execute("SELECT ts_utc, payload_json FROM kv_cache WHERE k=?", (key,))
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    ts_utc, payload = row
    if int(time.time()) - int(ts_utc) > int(ttl_sec):
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


# ============================================================
# yfinance wrappers with backoff + cache + single-session
# ============================================================
@st.cache_resource
def get_yf_session():
    # yfinance internally uses requests; caching is handled by our sqlite layer,
    # but a single session reduces overhead.
    return yf.Ticker

def _sleep_backoff(attempt: int):
    # exponential backoff with jitter
    base = 1.2 ** attempt
    time.sleep(min(10.0, base + random.random()))

def yf_history_cached(ticker: str, period: str, interval: str, ttl_sec: int):
    key = cache_key("history", ticker, period, interval)
    cached = cache_get(key, ttl_sec)
    if cached:
        df = pd.DataFrame(cached["rows"])
        # restore datetime index if present
        if "Date" in df.columns:
            try:
                df["Date"] = pd.to_datetime(df["Date"])
            except Exception:
                pass
        return df

    last_err = None
    for i in range(MAX_RETRIES):
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period=period, interval=interval)
            if hist is None or hist.empty:
                raise RuntimeError("No price data returned.")
            df = hist.reset_index()
            # serialize minimal
            cache_set(key, {"rows": df.to_dict(orient="records")})
            return df
        except YFRateLimitError as e:
            last_err = e
            _sleep_backoff(i + 1)
        except Exception as e:
            last_err = e
            _sleep_backoff(i + 1)

    # If rate-limited or failing, try stale (ignore TTL) as a last resort:
    stale = cache_get(key, ttl_sec=10**9)
    if stale:
        df = pd.DataFrame(stale["rows"])
        if "Date" in df.columns:
            try:
                df["Date"] = pd.to_datetime(df["Date"])
            except Exception:
                pass
        return df

    raise last_err if last_err else RuntimeError("Failed to fetch price history.")

def yf_expiries_cached(ticker: str, ttl_sec: int):
    key = cache_key("expiries", ticker)
    cached = cache_get(key, ttl_sec)
    if cached:
        return cached["expiries"]

    last_err = None
    for i in range(MAX_RETRIES):
        try:
            t = yf.Ticker(ticker)
            exps = t.options
            if not exps:
                raise RuntimeError("No expiries returned.")
            cache_set(key, {"expiries": list(exps)})
            return list(exps)
        except YFRateLimitError as e:
            last_err = e
            _sleep_backoff(i + 1)
        except Exception as e:
            last_err = e
            _sleep_backoff(i + 1)

    stale = cache_get(key, ttl_sec=10**9)
    if stale:
        return stale["expiries"]

    raise last_err if last_err else RuntimeError("Failed to fetch expiries.")

def yf_chain_cached(ticker: str, expiry: str, ttl_sec: int):
    key = cache_key("chain", ticker, expiry)
    cached = cache_get(key, ttl_sec)
    if cached:
        df = pd.DataFrame(cached["rows"])
        return df

    last_err = None
    for i in range(MAX_RETRIES):
        try:
            t = yf.Ticker(ticker)
            oc = t.option_chain(expiry)
            calls = oc.calls.copy()
            puts = oc.puts.copy()
            calls["option_type"] = "call"
            puts["option_type"] = "put"
            df = pd.concat([calls, puts], ignore_index=True)
            df["expiry"] = pd.to_datetime(expiry).date()
            cache_set(key, {"rows": df.to_dict(orient="records")})
            return df
        except YFRateLimitError as e:
            last_err = e
            _sleep_backoff(i + 1)
        except Exception as e:
            last_err = e
            _sleep_backoff(i + 1)

    stale = cache_get(key, ttl_sec=10**9)
    if stale:
        return pd.DataFrame(stale["rows"])

    raise last_err if last_err else RuntimeError("Failed to fetch option chain.")


# ============================================================
# Math (no scipy)
# ============================================================
SQRT_2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT_2))

def norm_pdf(x: float) -> float:
    return INV_SQRT_2PI * math.exp(-0.5 * x * x)

def bs_d1_d2(S, K, T, r, q, sigma):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return None, None
    vs = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vs
    d2 = d1 - vs
    return d1, d2

def bs_delta(S, K, T, r, q, sigma, opt_type):
    d1, _ = bs_d1_d2(S, K, T, r, q, sigma)
    if d1 is None:
        return None
    if opt_type == "call":
        return math.exp(-q*T) * norm_cdf(d1)
    return -math.exp(-q*T) * norm_cdf(-d1)

def bs_gamma(S, K, T, r, q, sigma):
    d1, _ = bs_d1_d2(S, K, T, r, q, sigma)
    if d1 is None:
        return None
    return (math.exp(-q*T) * norm_pdf(d1)) / (S * sigma * math.sqrt(T))

def charm_fd(S, K, T, r, q, sigma, opt_type, dt_days=1):
    if T <= 0:
        return None
    dt = dt_days / 365.0
    T2 = max(T - dt, 1e-6)
    d_now = bs_delta(S, K, T, r, q, sigma, opt_type)
    d_next = bs_delta(S, K, T2, r, q, sigma, opt_type)
    if d_now is None or d_next is None:
        return None
    return (d_next - d_now) / dt

def prob_finish_beyond(S0, L, T, r, q, sigma, direction: str):
    if T <= 0 or sigma <= 0 or S0 <= 0 or L <= 0:
        return None
    mu = (r - q - 0.5*sigma*sigma)*T
    denom = sigma * math.sqrt(T)
    z = (math.log(L / S0) - mu) / denom
    return (1.0 - norm_cdf(z)) if direction == "above" else norm_cdf(z)

def prob_touch_barrier(S0, B, T, r, q, sigma, barrier_type: str):
    if T <= 0 or sigma <= 0 or S0 <= 0 or B <= 0:
        return None
    if barrier_type == "up" and B <= S0:
        return 1.0
    if barrier_type == "down" and B >= S0:
        return 1.0
    drift = (r - q - 0.5*sigma*sigma)
    denom = sigma * math.sqrt(T)
    x = math.log(B / S0)
    term1 = norm_cdf(-(x - drift*T)/denom)
    term2 = math.exp(2*drift*x/(sigma*sigma)) * norm_cdf(-(x + drift*T)/denom)
    p = term1 + term2
    return max(0.0, min(1.0, float(p)))


# ============================================================
# Normalization + analytics
# ============================================================
def realized_vol(hist_df: pd.DataFrame, window=21):
    if hist_df is None or hist_df.empty:
        return None
    col = "Close" if "Close" in hist_df.columns else None
    if not col:
        return None
    c = pd.to_numeric(hist_df[col], errors="coerce").dropna()
    if len(c) < window + 2:
        return None
    rets = np.log(c).diff().dropna()
    return float(rets.tail(window).std() * math.sqrt(252))

def bucket_label(dte: int) -> str:
    if dte <= SHORT_DAYS:
        return "Short (≤14D)"
    if dte <= MID_DAYS:
        return "Mid (15–180D)"
    return "Long (≥181D)"

def normalize_all_chains(ticker: str, expiries: list[str], spot: float, ttl_sec: int) -> pd.DataFrame:
    today = date.today()
    frames = []
    for ex in expiries:
        df = yf_chain_cached(ticker, ex, ttl_sec=ttl_sec)
        exp_date = pd.to_datetime(ex).date()
        df["dte"] = int((exp_date - today).days)
        frames.append(df)

    big = pd.concat(frames, ignore_index=True)

    big["strike"] = pd.to_numeric(big["strike"], errors="coerce")
    big["volume"] = pd.to_numeric(big.get("volume", 0), errors="coerce").fillna(0).astype(int)
    big["openInterest"] = pd.to_numeric(big.get("openInterest", 0), errors="coerce").fillna(0).astype(int)
    big["impliedVolatility"] = pd.to_numeric(big.get("impliedVolatility", np.nan), errors="coerce")

    big = big.dropna(subset=["strike"])
    big = big[big["dte"] >= 1].copy()
    big["bucket"] = big["dte"].apply(bucket_label)
    big["moneyness"] = big["strike"] / float(spot)
    return big

def pick_atm_iv(chain: pd.DataFrame, spot: float, target_days: int):
    d = chain.copy()
    d["dte_diff"] = (d["dte"] - target_days).abs()
    if d.empty:
        return None, None, None
    exp = d.sort_values("dte_diff").iloc[0]["expiry"]
    e = d[d["expiry"] == exp].copy()
    e["k_diff"] = (e["strike"] - spot).abs()
    atm_k = float(e.sort_values("k_diff").iloc[0]["strike"])
    atm = e[e["strike"] == atm_k]
    ivs = atm["impliedVolatility"].dropna().tolist()
    ivs = [float(x) for x in ivs if x and x > 0]
    if not ivs:
        return None, exp, atm_k
    return float(np.mean(ivs)), exp, atm_k

def strike_3bar_frame(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    calls = df[df["option_type"] == "call"].groupby("strike", as_index=False)[metric].sum().rename(columns={metric: "Call"})
    puts  = df[df["option_type"] == "put"].groupby("strike", as_index=False)[metric].sum().rename(columns={metric: "Put"})
    out = pd.merge(calls, puts, on="strike", how="outer").fillna(0)
    out["Total"] = out["Call"] + out["Put"]
    out = out.sort_values("strike")
    for c in ["Call", "Put", "Total"]:
        out[c] = out[c].round(0).astype(int)
    return out


# ============================================================
# Snapshot DB (screener)
# ============================================================
def db_init():
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            snapshot_id TEXT PRIMARY KEY,
            ts_utc TEXT,
            ticker TEXT,
            spot REAL,
            spot_ts TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chain_rows (
            snapshot_id TEXT,
            expiry TEXT,
            dte INTEGER,
            option_type TEXT,
            strike REAL,
            volume INTEGER,
            open_interest INTEGER
        )
    """)
    con.commit()
    con.close()

def snapshot_store(ticker: str, spot: float, spot_ts: str, chain: pd.DataFrame) -> str:
    db_init()
    sid = f"{ticker}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO snapshots VALUES (?,?,?,?,?)",
        (sid, datetime.utcnow().isoformat(), ticker, float(spot), str(spot_ts))
    )
    rows = []
    for _, r in chain[["expiry", "dte", "option_type", "strike", "volume", "openInterest"]].iterrows():
        rows.append((sid, str(r["expiry"]), int(r["dte"]), str(r["option_type"]), float(r["strike"]), int(r["volume"]), int(r["openInterest"])))
    cur.executemany("INSERT INTO chain_rows VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return sid

def get_last_two_snapshots(ticker: str):
    db_init()
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute(
        "SELECT snapshot_id, ts_utc, spot FROM snapshots WHERE ticker=? ORDER BY ts_utc DESC LIMIT 2",
        (ticker,)
    )
    rows = cur.fetchall()
    con.close()
    return rows

def screener_watchlist(tickers, price_move_max_pct=1.0, oi_jump_min=5000, vol_jump_min=5000):
    out = []
    db_init()
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()

    for tkr in tickers:
        snaps = get_last_two_snapshots(tkr)
        if len(snaps) < 2:
            continue

        s0, ts0, spot0 = snaps[0]
        s1, ts1, spot1 = snaps[1]
        if spot1 == 0:
            continue

        chg = (spot0 - spot1) / spot1 * 100.0
        if abs(chg) > price_move_max_pct:
            continue

        cur.execute(
            "SELECT option_type, SUM(open_interest), SUM(volume) FROM chain_rows WHERE snapshot_id=? GROUP BY option_type",
            (s0,)
        )
        new = {r[0]: (r[1] or 0, r[2] or 0) for r in cur.fetchall()}

        cur.execute(
            "SELECT option_type, SUM(open_interest), SUM(volume) FROM chain_rows WHERE snapshot_id=? GROUP BY option_type",
            (s1,)
        )
        old = {r[0]: (r[1] or 0, r[2] or 0) for r in cur.fetchall()}

        for side in ["call", "put"]:
            oi_new, vol_new = new.get(side, (0, 0))
            oi_old, vol_old = old.get(side, (0, 0))
            oi_jump = oi_new - oi_old
            vol_jump = vol_new - vol_old

            if (oi_jump >= oi_jump_min) or (vol_jump >= vol_jump_min):
                out.append({
                    "ticker": tkr,
                    "side": side,
                    "spot_change_pct": chg,
                    "oi_jump_total": oi_jump,
                    "vol_jump_total": vol_jump,
                    "new_snapshot_utc": ts0,
                    "old_snapshot_utc": ts1,
                })

    con.close()
    return pd.DataFrame(out)


# ============================================================
# Plotly helpers
# ============================================================
def style_fig(fig, title: str):
    fig.update_layout(
        template="plotly_dark",
        height=520,
        title=title,
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(color="#E6E6E6", size=13),
        margin=dict(l=10, r=10, t=70, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
    )
    return fig

def add_spot_line(fig, spot: float, ymax: float):
    fig.add_vline(x=spot, line_width=2, line_dash="dash", line_color="#FF991C")
    fig.add_annotation(
        x=spot, y=ymax,
        xref="x", yref="y",
        text=f"Spot {spot:,.2f}",
        showarrow=True,
        arrowhead=2,
        ax=30, ay=-40,
        font=dict(color="#FF991C", size=13),
        bgcolor="rgba(0,0,0,0.65)",
        bordercolor="rgba(255,153,28,0.8)",
        borderwidth=1
    )
    return fig

def plot_3bar(df3: pd.DataFrame, metric: str, spot_val: float, title: str):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df3["strike"], y=df3["Call"], name="Call", marker_color="#00B3FF"))
    fig.add_trace(go.Bar(x=df3["strike"], y=df3["Put"],  name="Put",  marker_color="#FF2DAA"))
    fig.add_trace(go.Bar(x=df3["strike"], y=df3["Total"],name="Total",marker_color="#FF991C"))
    fig.update_layout(barmode="group")
    ymax = max(1, int(df3[["Call", "Put", "Total"]].to_numpy().max()))
    fig = add_spot_line(fig, spot_val, ymax)
    fig = style_fig(fig, title)
    fig.update_yaxes(title_text=metric)
    fig.update_xaxes(title_text="Strike")
    return fig


# ============================================================
# UI Controls
# ============================================================
st.sidebar.markdown("## Controls")
ticker = st.sidebar.text_input("Ticker", value="SPY").upper().strip()

q = st.sidebar.number_input("Dividend yield q (decimal)", value=0.0, min_value=0.0, max_value=0.25, step=0.001, format="%.3f")

ttl_min = st.sidebar.slider("Cloud cache TTL (minutes)", 5, 60, int(DEFAULT_CACHE_TTL_SEC / 60), 5)
ttl_sec = int(ttl_min) * 60

watchlist_text = st.sidebar.text_area("Screener watchlist (comma or newline)", value="SPY\nQQQ\nAAPL\nMSFT")
watchlist = [x.strip().upper() for x in watchlist_text.replace(",", "\n").splitlines() if x.strip()]

if "last_refresh_utc" not in st.session_state:
    st.session_state.last_refresh_utc = 0
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

def can_refresh():
    return (int(time.time()) - int(st.session_state.last_refresh_utc)) >= MIN_REFRESH_INTERVAL_SEC

c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
refresh_clicked = c1.button("Refresh Data", disabled=not can_refresh())
snap_btn = c2.button("Take Snapshot")
screen_btn = c3.button("Run Screener")
c4.caption(f"Min refresh interval: {MIN_REFRESH_INTERVAL_SEC}s | TTL: {ttl_min}m")

if refresh_clicked or (not st.session_state.data_loaded):
    st.session_state.last_refresh_utc = int(time.time())
    st.session_state.data_loaded = True

# ============================================================
# DATA LOAD (only when refresh or first load)
# ============================================================
if "spot" not in st.session_state or refresh_clicked:
    try:
        hist = yf_history_cached(ticker, period="2y", interval="1d", ttl_sec=ttl_sec)
        spot = float(hist["Close"].iloc[-1])
        spot_ts = str(hist["Date"].iloc[-1]) if "Date" in hist.columns else "NA"

        expiries = yf_expiries_cached(ticker, ttl_sec=ttl_sec)
        # NOTE: fetching ALL expiries is expensive. Still supported, but cached.
        chain_raw = normalize_all_chains(ticker, expiries, spot, ttl_sec=ttl_sec)

        st.session_state["hist"] = hist
        st.session_state["spot"] = spot
        st.session_state["spot_ts"] = spot_ts
        st.session_state["expiries"] = expiries
        st.session_state["chain"] = chain_raw

    except YFRateLimitError:
        st.error(
            "Yahoo rate limit hit. This app will automatically use the most recent cached data if available. "
            "Wait a few minutes and retry Refresh."
        )
    except Exception as e:
        st.error(f"Data load failed: {e}")

# If we still have no data, stop cleanly (no crashing UI)
if "spot" not in st.session_state or "chain" not in st.session_state:
    st.stop()

spot = st.session_state["spot"]
spot_ts = st.session_state["spot_ts"]
hist = st.session_state["hist"]
expiries = st.session_state["expiries"]
chain = st.session_state["chain"]

hv21 = realized_vol(hist, 21)
hv63 = realized_vol(hist, 63)

# Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Spot", f"{spot:,.2f}")
m2.metric("HV(21)", f"{hv21*100:.1f}%" if hv21 else "NA")
m3.metric("HV(63)", f"{hv63*100:.1f}%" if hv63 else "NA")
m4.metric("Expiries", f"{len(expiries)}")
st.caption(f"Spot timestamp: {spot_ts} (cached + throttled)")

if snap_btn:
    sid = snapshot_store(ticker, spot, spot_ts, chain)
    st.success(f"Snapshot stored: {sid}")


# ============================================================
# Tabs
# ============================================================
tabs = st.tabs([
    "Aggregated OI/Volume",
    "Expiry Slice",
    "S/R + Probability",
    "Gamma Wall + Charm",
    "Vol Surface",
    "Abnormal Activity Screener",
    "Report Builder",
])

# ============================================================
# TAB 1
# ============================================================
with tabs[0]:
    st.subheader("Aggregated by Strike (All Expiries Combined)")
    st.caption("Per strike: Call / Put / Total shown concurrently.")

    a1, a2, a3, a4, a5 = st.columns([1, 1, 1, 1, 1])
    metric = a1.selectbox("Metric", ["openInterest", "volume"], index=0)
    scope = a2.selectbox("Expiry bucket", ["All", "Short (≤14D)", "Mid (15–180D)", "Long (≥181D)"], index=0)
    topn = a3.slider("Top strikes (table)", 10, 80, 30, 5)
    use_window = a4.selectbox("Strike filter", ["Around spot", "All strikes"], index=0)
    window_pct = a5.slider("±% of spot", 5, 100, 25, 5, disabled=(use_window != "Around spot"))

    d = chain.copy()
    if scope != "All":
        d = d[d["bucket"] == scope]
    if use_window == "Around spot":
        lo = spot * (1 - window_pct / 100)
        hi = spot * (1 + window_pct / 100)
        d = d[(d["strike"] >= lo) & (d["strike"] <= hi)]

    df3 = strike_3bar_frame(d, metric)
    st.dataframe(df3.sort_values("Total", ascending=False).head(topn), use_container_width=True, height=260)
    st.plotly_chart(plot_3bar(df3, metric, spot, f"{metric} by Strike (Aggregated) — Call/Put/Total"), use_container_width=True)

# ============================================================
# TAB 2
# ============================================================
with tabs[1]:
    st.subheader("Per-Expiry Accumulation by Strike")
    metric2 = st.selectbox("Metric", ["openInterest", "volume"], index=0, key="metric2")
    exp_sel = st.selectbox("Expiry", sorted(chain["expiry"].unique().tolist()))

    use_window2 = st.selectbox("Strike filter", ["Around spot", "All strikes"], index=0, key="win2")
    window_pct2 = st.slider("±% of spot", 5, 100, 25, 5, key="wp2", disabled=(use_window2 != "Around spot"))

    d = chain[chain["expiry"] == exp_sel].copy()
    if use_window2 == "Around spot":
        lo = spot * (1 - window_pct2 / 100)
        hi = spot * (1 + window_pct2 / 100)
        d = d[(d["strike"] >= lo) & (d["strike"] <= hi)]

    df3e = strike_3bar_frame(d, metric2)
    st.dataframe(df3e.sort_values("Total", ascending=False).head(40), use_container_width=True, height=260)
    st.plotly_chart(plot_3bar(df3e, metric2, spot, f"{metric2} by Strike @ {exp_sel} — Call/Put/Total"), use_container_width=True)

# ============================================================
# TAB 3
# ============================================================
with tabs[2]:
    st.subheader("Support/Resistance by Horizon + Black–Scholes Probabilities")
    st.caption("Short=14D, Mid=180D, Long=365D (deterministic). Uses chain ATM IV when available, otherwise real HV fallback.")

    strength_metric = st.selectbox("Strength metric", ["openInterest", "volume"], index=0)
    side3 = st.selectbox("Use strikes from", ["both", "put", "call"], index=0)
    hv_fallback = hv63 or hv21

    def top_levels_for_bucket(bucket_name: str, days: int, n=3):
        subset = chain[chain["bucket"] == bucket_name].copy()
        if side3 in ("call", "put"):
            subset = subset[subset["option_type"] == side3]

        puts = subset[subset["option_type"] == "put"]
        calls = subset[subset["option_type"] == "call"]

        sup = puts[puts["strike"] <= spot].groupby("strike", as_index=False)[strength_metric].sum()
        res = calls[calls["strike"] >= spot].groupby("strike", as_index=False)[strength_metric].sum()

        sup = sup.sort_values(strength_metric, ascending=False).head(n)
        res = res.sort_values(strength_metric, ascending=False).head(n)

        iv_atm, exp_used, atm_k = pick_atm_iv(chain, spot, days)
        sigma = iv_atm if (iv_atm and iv_atm > 0) else hv_fallback
        sigma_src = "ATM IV (real chain)" if (iv_atm and iv_atm > 0) else "HV fallback (real underlying)"
        T = days / 365.0

        def add_probs(df, kind):
            if df.empty or not sigma:
                df["P_touch"] = np.nan
                df["P_finish"] = np.nan
                return df
            pt, pf = [], []
            for _, rr in df.iterrows():
                L = float(rr["strike"])
                if kind == "support":
                    pt.append(prob_touch_barrier(spot, L, T, r=0.0, q=q, sigma=float(sigma), barrier_type="down"))
                    pf.append(prob_finish_beyond(spot, L, T, r=0.0, q=q, sigma=float(sigma), direction="below"))
                else:
                    pt.append(prob_touch_barrier(spot, L, T, r=0.0, q=q, sigma=float(sigma), barrier_type="up"))
                    pf.append(prob_finish_beyond(spot, L, T, r=0.0, q=q, sigma=float(sigma), direction="above"))
            df["P_touch"] = pt
            df["P_finish"] = pf
            return df

        return sigma, sigma_src, exp_used, atm_k, add_probs(sup, "support"), add_probs(res, "resistance")

    for bucket_name, days in [("Short (≤14D)", SHORT_DAYS), ("Mid (15–180D)", MID_DAYS), ("Long (≥181D)", LONG_DAYS)]:
        st.markdown("---")
        sigma, sigma_src, exp_used, atm_k, sup, res = top_levels_for_bucket(bucket_name, days, n=3)
        st.markdown(f"### {bucket_name}")
        st.caption(f"σ source: {sigma_src} | ATM expiry used: {exp_used} | ATM strike: {atm_k} | q={q:.3f}")

        cL, cR = st.columns(2)
        with cL:
            st.markdown("**Support**")
            st.dataframe(sup, use_container_width=True, height=160)
        with cR:
            st.markdown("**Resistance**")
            st.dataframe(res, use_container_width=True, height=160)

# ============================================================
# TAB 4
# ============================================================
with tabs[3]:
    st.subheader("Gamma Wall + Charm (IV-required; no fake IV)")
    side4 = st.selectbox("Side (gamma/charm slice)", ["both", "call", "put"], index=0, key="side4")
    scope4 = st.selectbox("Expiry bucket (gamma/charm)", ["All", "Short (≤14D)", "Mid (15–180D)", "Long (≥181D)"], index=0, key="scope4")

    d = chain.copy()
    if scope4 != "All":
        d = d[d["bucket"] == scope4]
    if side4 in ("call", "put"):
        d = d[d["option_type"] == side4]

    d = d[d["impliedVolatility"].notna() & (d["impliedVolatility"] > 0)].copy()
    if d.empty:
        st.warning("No IV available for this slice; gamma/charm cannot be computed without real IV.")
    else:
        d["T"] = d["dte"] / 365.0
        d["gamma"] = [
            (bs_gamma(spot, float(r["strike"]), float(r["T"]), r=0.0, q=q, sigma=float(r["impliedVolatility"])) or np.nan)
            for _, r in d.iterrows()
        ]
        d["gex_proxy"] = d["gamma"] * d["openInterest"] * CONTRACT_MULTIPLIER * (spot ** 2)

        agg_gex = d.groupby("strike", as_index=False)["gex_proxy"].sum().sort_values("strike")
        fig4 = px.bar(agg_gex, x="strike", y="gex_proxy")
        fig4.update_traces(marker_color="#FF991C")
        ymax = max(1.0, float(np.nanmax(agg_gex["gex_proxy"].values)))
        fig4 = add_spot_line(fig4, spot, ymax)
        fig4 = style_fig(fig4, "Gamma Exposure Proxy by Strike")
        st.plotly_chart(fig4, use_container_width=True)

        topwalls = agg_gex.reindex(agg_gex["gex_proxy"].abs().sort_values(ascending=False).index).head(15)
        st.dataframe(topwalls, use_container_width=True, height=240)

# ============================================================
# TAB 5
# ============================================================
with tabs[4]:
    st.subheader("Implied Volatility Surface (Real Chain IV)")
    d = chain[chain["impliedVolatility"].notna() & (chain["impliedVolatility"] > 0)].copy()
    if d.empty:
        st.warning("No IV data available from chain for this ticker.")
    else:
        grid = d.groupby(["expiry", "strike"], as_index=False)["impliedVolatility"].mean()
        piv = grid.pivot_table(index="expiry", columns="strike", values="impliedVolatility", aggfunc="mean").sort_index()

        figH = go.Figure(data=go.Heatmap(
            z=piv.values,
            x=piv.columns.astype(float),
            y=[str(x) for x in piv.index],
            colorbar=dict(title="IV"),
            colorscale="Viridis"
        ))
        figH.update_layout(
            template="plotly_dark",
            height=560,
            title="IV Heatmap: Expiry × Strike",
            paper_bgcolor="#000000",
            plot_bgcolor="#000000",
            font=dict(color="#E6E6E6"),
            margin=dict(l=10, r=10, t=70, b=10),
        )
        st.plotly_chart(figH, use_container_width=True)

# ============================================================
# TAB 6
# ============================================================
with tabs[5]:
    st.subheader("Abnormal Options Activity Screener (Snapshot-based; Real Only)")
    st.caption("Take snapshots over time. This does not fabricate historical options chains.")

    colA, colB, colC = st.columns(3)
    price_move_max = colA.number_input("Max abs spot move %", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    oi_jump_min = colB.number_input("Min total OI jump (per side)", min_value=100.0, max_value=500000.0, value=5000.0, step=100.0)
    vol_jump_min = colC.number_input("Min total Volume jump (per side)", min_value=100.0, max_value=500000.0, value=5000.0, step=100.0)

    if screen_btn:
        out = screener_watchlist(watchlist, float(price_move_max), float(oi_jump_min), float(vol_jump_min))
        if out.empty:
            st.info("No flags (or not enough snapshots yet). Take 2+ snapshots per ticker.")
        else:
            st.dataframe(out.sort_values(["oi_jump_total", "vol_jump_total"], ascending=False),
                         use_container_width=True, height=420)

# ============================================================
# TAB 7: REPORT BUILDER (new)
# ============================================================
with tabs[6]:
    st.subheader("Report Builder")
    st.caption("Select sections to include, then download a Markdown summary (real data only).")

    # Precompute some commonly-used summaries
    chain_now = chain.copy()

    report_sections = st.multiselect(
        "Include sections",
        [
            "Header (Spot/HV/Expiries)",
            "Aggregated OI (Top 15 strikes)",
            "Aggregated Volume (Top 15 strikes)",
            "Nearest Expiry Slice (Top 15 strikes)",
            "Support/Resistance + Probabilities (Short/Mid/Long)",
            "Gamma Wall (Top 10 |GEX|)",
            "IV Surface (notes only)"
        ],
        default=[
            "Header (Spot/HV/Expiries)",
            "Aggregated OI (Top 15 strikes)",
            "Support/Resistance + Probabilities (Short/Mid/Long)",
        ]
    )

    def md_table(df: pd.DataFrame, max_rows=20):
        if df is None or df.empty:
            return "_No data._"
        d = df.head(max_rows).copy()
        return d.to_markdown(index=False)

    def build_report_md() -> str:
        lines = []
        lines.append(f"# {APP_TITLE}")
        lines.append(f"**Ticker:** {ticker}")
        lines.append(f"**Generated (UTC):** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}  ")
        lines.append(f"**Data policy:** Real chain data via Yahoo Finance/yfinance; persistent cache TTL {ttl_min}m; no simulated data.")
        lines.append("")

        if "Header (Spot/HV/Expiries)" in report_sections:
            lines.append("## Market Snapshot")
            lines.append(f"- Spot: **{spot:,.2f}**")
            lines.append(f"- HV(21): **{(hv21*100):.2f}%**" if hv21 else "- HV(21): NA")
            lines.append(f"- HV(63): **{(hv63*100):.2f}%**" if hv63 else "- HV(63): NA")
            lines.append(f"- Expiries: **{len(expiries)}**")
            lines.append("")

        if "Aggregated OI (Top 15 strikes)" in report_sections:
            df3 = strike_3bar_frame(chain_now, "openInterest").sort_values("Total", ascending=False).head(15)
            lines.append("## Aggregated Open Interest (Top 15 strikes)")
            lines.append(md_table(df3, 15))
            lines.append("")

        if "Aggregated Volume (Top 15 strikes)" in report_sections:
            df3 = strike_3bar_frame(chain_now, "volume").sort_values("Total", ascending=False).head(15)
            lines.append("## Aggregated Volume (Top 15 strikes)")
            lines.append(md_table(df3, 15))
            lines.append("")

        if "Nearest Expiry Slice (Top 15 strikes)" in report_sections:
            nearest_exp = sorted(chain_now["expiry"].unique().tolist())[0]
            d = chain_now[chain_now["expiry"] == nearest_exp]
            df3 = strike_3bar_frame(d, "openInterest").sort_values("Total", ascending=False).head(15)
            lines.append(f"## Nearest Expiry Slice — {nearest_exp} (Top 15 OI strikes)")
            lines.append(md_table(df3, 15))
            lines.append("")

        if "Support/Resistance + Probabilities (Short/Mid/Long)" in report_sections:
            lines.append("## Support/Resistance + Probabilities")
            lines.append("_Method: Support from put concentration at/under spot; resistance from call concentration at/above spot. Probabilities use lognormal (Black–Scholes) assumptions with ATM IV when available; otherwise HV fallback._")
            lines.append("")

            strength_metric = "openInterest"
            hv_fallback = hv63 or hv21

            def top_levels(bucket_name: str, days: int, n=3):
                subset = chain_now[chain_now["bucket"] == bucket_name].copy()
                puts = subset[subset["option_type"] == "put"]
                calls = subset[subset["option_type"] == "call"]

                sup = puts[puts["strike"] <= spot].groupby("strike", as_index=False)[strength_metric].sum().sort_values(strength_metric, ascending=False).head(n)
                res = calls[calls["strike"] >= spot].groupby("strike", as_index=False)[strength_metric].sum().sort_values(strength_metric, ascending=False).head(n)

                iv_atm, exp_used, atm_k = pick_atm_iv(chain_now, spot, days)
                sigma = iv_atm if (iv_atm and iv_atm > 0) else hv_fallback
                T = days / 365.0

                def add_probs(df, kind):
                    if df.empty or not sigma:
                        df["P_touch"] = np.nan
                        df["P_finish"] = np.nan
                        return df
                    pt, pf = [], []
                    for _, rr in df.iterrows():
                        L = float(rr["strike"])
                        if kind == "support":
                            pt.append(prob_touch_barrier(spot, L, T, r=0.0, q=q, sigma=float(sigma), barrier_type="down"))
                            pf.append(prob_finish_beyond(spot, L, T, r=0.0, q=q, sigma=float(sigma), direction="below"))
                        else:
                            pt.append(prob_touch_barrier(spot, L, T, r=0.0, q=q, sigma=float(sigma), barrier_type="up"))
                            pf.append(prob_finish_beyond(spot, L, T, r=0.0, q=q, sigma=float(sigma), direction="above"))
                    df["P_touch"] = [float(x) if x is not None else np.nan for x in pt]
                    df["P_finish"] = [float(x) if x is not None else np.nan for x in pf]
                    return df

                return exp_used, atm_k, sigma, add_probs(sup, "support"), add_probs(res, "resistance")

            for bucket_name, days in [("Short (≤14D)", SHORT_DAYS), ("Mid (15–180D)", MID_DAYS), ("Long (≥181D)", LONG_DAYS)]:
                exp_used, atm_k, sigma, sup, res = top_levels(bucket_name, days, n=3)
                lines.append(f"### {bucket_name}")
                lines.append(f"- ATM expiry used for σ: {exp_used} | ATM strike: {atm_k} | σ: {sigma if sigma else 'NA'} | q={q:.3f}")
                lines.append("")
                lines.append("**Support**")
                lines.append(md_table(sup, 10))
                lines.append("")
                lines.append("**Resistance**")
                lines.append(md_table(res, 10))
                lines.append("")

        if "Gamma Wall (Top 10 |GEX|)" in report_sections:
            d = chain_now.copy()
            d = d[d["impliedVolatility"].notna() & (d["impliedVolatility"] > 0)].copy()
            if d.empty:
                lines.append("## Gamma Wall")
                lines.append("_No IV available; cannot compute gamma-based measures._")
                lines.append("")
            else:
                d["T"] = d["dte"] / 365.0
                d["gamma"] = [
                    (bs_gamma(spot, float(r["strike"]), float(r["T"]), r=0.0, q=q, sigma=float(r["impliedVolatility"])) or np.nan)
                    for _, r in d.iterrows()
                ]
                d["gex_proxy"] = d["gamma"] * d["openInterest"] * CONTRACT_MULTIPLIER * (spot ** 2)
                agg = d.groupby("strike", as_index=False)["gex_proxy"].sum()
                top = agg.reindex(agg["gex_proxy"].abs().sort_values(ascending=False).index).head(10)
                lines.append("## Gamma Wall (Top 10 |GEX proxy|)")
                lines.append(md_table(top, 10))
                lines.append("")

        if "IV Surface (notes only)" in report_sections:
            lines.append("## IV Surface")
            lines.append("- View the IV heatmap in the Vol Surface tab (derived from real chain implied vol).")
            lines.append("")

        return "\n".join(lines)

    report_md = build_report_md()
    st.text_area("Preview (Markdown)", report_md, height=420)

    st.download_button(
        label="Download report as .md",
        data=report_md.encode("utf-8"),
        file_name=f"{ticker}_options_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}Z.md",
        mime="text/markdown"
    )

st.markdown("---")
st.caption("Rate-limit fix: persistent sqlite cache + backoff + fetch-on-refresh (no rerun spamming). Report Builder: selectable sections + downloadable Markdown.")
