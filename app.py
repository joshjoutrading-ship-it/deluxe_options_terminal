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
            cache_set(key, {"expiries": list(exps)})
            return list(exps)
        except YFRateLimitError as e:
            last_err = e
            _sleep_backoff(i + 1)
        except Exception as e:
            last_err = e
            _sleep_backoff(i + 1)

    stale = cache_get(key, ttl_sec=ttl_sec, allow_stale=True)
    if stale:
        return stale["expiries"]
    raise last_err if last_err else RuntimeError("Failed to fetch expiries.")

def yf_chain_cached(ticker: str, expiry: str, ttl_sec: int):
    key = cache_key("chain", ticker, expiry)
    cached = cache_get(key, ttl_sec=ttl_sec)
    if cached:
        return pd.DataFrame(cached["rows"])

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

    stale = cache_get(key, ttl_sec=ttl_sec, allow_stale=True)
    if stale:
        return pd.DataFrame(stale["rows"])
    raise last_err if last_err else RuntimeError("Failed to fetch chain.")

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

def bs_gamma(S, K, T, r, q, sigma):
    d1, _ = bs_d1_d2(S, K, T, r, q, sigma)
    if d1 is None:
        return None
    return (math.exp(-q*T) * norm_pdf(d1)) / (S * sigma * math.sqrt(T))

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
# Analytics helpers
# ============================================================
def realized_vol(hist_df: pd.DataFrame, window=21):
    if hist_df is None or hist_df.empty or "Close" not in hist_df.columns:
        return None
    c = pd.to_numeric(hist_df["Close"], errors="coerce").dropna()
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

def normalize_chain(df: pd.DataFrame, expiry: str, spot: float):
    today = date.today()
    exp_date = pd.to_datetime(expiry).date()
    df = df.copy()
    df["expiry"] = pd.to_datetime(expiry).date()
    df["dte"] = int((exp_date - today).days)

    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0).astype(int)
    df["openInterest"] = pd.to_numeric(df.get("openInterest", 0), errors="coerce").fillna(0).astype(int)
    df["impliedVolatility"] = pd.to_numeric(df.get("impliedVolatility", np.nan), errors="coerce")

    df = df.dropna(subset=["strike"])
    df = df[df["dte"] >= 1].copy()
    df["bucket"] = df["dte"].apply(bucket_label)
    df["moneyness"] = df["strike"] / float(spot)
    return df

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
# Sidebar controls
# ============================================================
st.sidebar.markdown("## Controls")
ticker = st.sidebar.text_input("Ticker", value="SPY").upper().strip()
q = st.sidebar.number_input("Dividend yield q (decimal)", value=0.0, min_value=0.0, max_value=0.25, step=0.001, format="%.3f")

st.sidebar.markdown("### Cloud rate-limit controls")
mode = st.sidebar.selectbox(
    "Data mode",
    ["Lean (recommended on Cloud)", "Full aggregate (heavy; may rate-limit)"],
    index=0
)

# In Lean mode, we cap option_chain calls
max_expiries_total = st.sidebar.slider("Max expiries to load (Lean mode)", 1, 30, 8, 1)
per_bucket = st.sidebar.slider("Per-bucket expiry cap (Lean mode)", 1, 10, 3, 1)

spot_ttl_min = st.sidebar.slider("Spot TTL (minutes)", 2, 60, int(SPOT_TTL_SEC_DEFAULT / 60), 1)
chain_ttl_hr = st.sidebar.slider("Options chain TTL (hours)", 1, 24, int(CHAIN_TTL_SEC_DEFAULT / 3600), 1)

spot_ttl = int(spot_ttl_min) * 60
exp_ttl = int(EXPIRIES_TTL_SEC_DEFAULT)
chain_ttl = int(chain_ttl_hr) * 3600

watchlist_text = st.sidebar.text_area("Screener watchlist (comma or newline)", value="SPY\nQQQ\nAAPL\nMSFT")
watchlist = [x.strip().upper() for x in watchlist_text.replace(",", "\n").splitlines() if x.strip()]

if "last_refresh_utc" not in st.session_state:
    st.session_state.last_refresh_utc = 0

def can_refresh():
    return (int(time.time()) - int(st.session_state.last_refresh_utc)) >= MIN_REFRESH_INTERVAL_SEC

c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
refresh_clicked = c1.button("Refresh Data", disabled=not can_refresh())
build_full_clicked = c2.button("Build Full Aggregate Cache", disabled=(mode != "Lean (recommended on Cloud)"))
c3.button("Run Screener")  # placeholder (kept simple here)
c4.caption(f"Min refresh interval: {MIN_REFRESH_INTERVAL_SEC}s")

if refresh_clicked:
    st.session_state.last_refresh_utc = int(time.time())

# ============================================================
# Fetch spot + expiries (cheap-ish)
# ============================================================
try:
    hist = yf_history_cached(ticker, period="2y", interval="1d", ttl_sec=spot_ttl)
    spot = float(hist["Close"].iloc[-1])
    spot_ts = str(hist["Date"].iloc[-1]) if "Date" in hist.columns else "NA"
    expiries = yf_expiries_cached(ticker, ttl_sec=exp_ttl)
except YFRateLimitError:
    st.error("Rate-limited by Yahoo. If this is first run, you must wait. If you have cached data, it will load automatically.")
    st.stop()
except Exception as e:
    st.error(f"Failed to fetch spot/expiries: {e}")
    st.stop()

hv21 = realized_vol(hist, 21)
hv63 = realized_vol(hist, 63)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Spot", f"{spot:,.2f}")
m2.metric("HV(21)", f"{hv21*100:.1f}%" if hv21 else "NA")
m3.metric("HV(63)", f"{hv63*100:.1f}%" if hv63 else "NA")
m4.metric("Expiries", f"{len(expiries)}")
st.caption(f"Spot timestamp: {spot_ts}")

# ============================================================
# Expiry selection (always available)
# ============================================================
all_exp_dates = [pd.to_datetime(x).date() for x in expiries]
min_exp = min(all_exp_dates)
selected_exp = st.selectbox("Expiry (used for Expiry Slice + default analytics)", sorted(all_exp_dates))

# ============================================================
# Decide which expiries to load (critical rate-limit fix)
# ============================================================
today = date.today()

def expiry_dte(exp: date) -> int:
    return int((exp - today).days)

def bucket_for_dte(dte: int) -> str:
    if dte <= SHORT_DAYS:
        return "Short (≤14D)"
    if dte <= MID_DAYS:
        return "Mid (15–180D)"
    return "Long (≥181D)"

exp_df = pd.DataFrame({
    "expiry": sorted(all_exp_dates),
})
exp_df["dte"] = exp_df["expiry"].apply(expiry_dte)
exp_df["bucket"] = exp_df["dte"].apply(bucket_for_dte)

def pick_lean_expiries():
    picks = []
    for b in ["Short (≤14D)", "Mid (15–180D)", "Long (≥181D)"]:
        sub = exp_df[exp_df["bucket"] == b].sort_values("dte")
        picks.extend(sub["expiry"].head(per_bucket).tolist())
    # always include selected expiry even if outside caps
    if selected_exp not in picks:
        picks.append(selected_exp)
    # cap total
    picks = sorted(list(dict.fromkeys(picks)))  # unique preserving
    # If too many, keep closest to today plus selected
    if len(picks) > max_expiries_total:
        closest = exp_df.sort_values("dte")["expiry"].head(max_expiries_total).tolist()
        if selected_exp not in closest:
            closest[-1] = selected_exp
        picks = sorted(list(dict.fromkeys(closest)))
    return picks

if mode == "Lean (recommended on Cloud)":
    expiries_to_load = pick_lean_expiries()
else:
    expiries_to_load = sorted(all_exp_dates)

st.caption(f"Loaded expiries this run: {len(expiries_to_load)} (mode={mode})")

# ============================================================
# Load chains for chosen expiries (bounded in Lean mode)
# ============================================================
chains = []
rate_limited = False
loaded_count = 0

progress = st.progress(0, text="Loading option chains (cached where possible)...")

for idx, exp in enumerate(expiries_to_load):
    exp_str = str(exp)
    try:
        df = yf_chain_cached(ticker, exp_str, ttl_sec=chain_ttl)
        df = normalize_chain(df, exp_str, spot)
        chains.append(df)
        loaded_count += 1
    except YFRateLimitError:
        rate_limited = True
        break
    except Exception:
        # skip problematic expiry
        pass
    progress.progress(int((idx + 1) / max(1, len(expiries_to_load)) * 100))

progress.empty()

if not chains:
    st.error("No option chain data available (rate-limited or empty). Try again later.")
    st.stop()

chain = pd.concat(chains, ignore_index=True)

if rate_limited:
    st.warning("Rate limit hit while loading expiries. Displaying partial (cached/loaded) aggregate. Use Lean mode and/or wait before refreshing.")

# Optional: progressive full aggregate cache builder
if build_full_clicked:
    st.info("Building full aggregate cache progressively. This will attempt to fetch missing expiries and stop if rate-limited.")
    missing = [e for e in sorted(all_exp_dates) if e not in expiries_to_load]
    built = 0
    for e in missing:
        try:
            _ = yf_chain_cached(ticker, str(e), ttl_sec=chain_ttl)  # will cache
            built += 1
            time.sleep(0.25)  # pacing
        except YFRateLimitError:
            st.warning(f"Rate-limited during cache build. Built {built} additional expiries. Retry later.")
            break
        except Exception:
            pass
    st.success(f"Cache build complete for this run. Added {built} expiries to cache (if not rate-limited).")

# ============================================================
# Tabs + report builder
# ============================================================
tabs = st.tabs([
    "Aggregated OI/Volume",
    "Expiry Slice",
    "S/R + Probability",
    "Gamma Wall",
    "Vol Surface",
    "Report Builder",
])

with tabs[0]:
    st.subheader("Aggregated by Strike")
    metric = st.selectbox("Metric", ["openInterest", "volume"], index=0)
    scope = st.selectbox("Expiry bucket", ["All", "Short (≤14D)", "Mid (15–180D)", "Long (≥181D)"], index=0)
    topn = st.slider("Top strikes (table)", 10, 80, 30, 5)

    d = chain.copy()
    if scope != "All":
        d = d[d["bucket"] == scope]

    df3 = strike_3bar_frame(d, metric)
    st.dataframe(df3.sort_values("Total", ascending=False).head(topn), use_container_width=True, height=260)
    st.plotly_chart(plot_3bar(df3, metric, spot, f"{metric} by Strike — Call/Put/Total"), use_container_width=True)

with tabs[1]:
    st.subheader("Expiry Slice")
    metric2 = st.selectbox("Metric", ["openInterest", "volume"], index=0, key="metric2")

    d = chain[chain["expiry"] == selected_exp].copy()
    df3e = strike_3bar_frame(d, metric2)
    st.dataframe(df3e.sort_values("Total", ascending=False).head(40), use_container_width=True, height=260)
    st.plotly_chart(plot_3bar(df3e, metric2, spot, f"{metric2} by Strike @ {selected_exp} — Call/Put/Total"), use_container_width=True)

with tabs[2]:
    st.subheader("Support/Resistance + Probabilities")
    st.caption("Note: probabilities depend on σ. This uses ATM IV from the loaded chains; if missing, falls back to HV(63)/HV(21).")

    strength_metric = st.selectbox("Strength metric", ["openInterest", "volume"], index=0)
    hv_fallback = hv63 or hv21

    def pick_atm_iv_from_loaded(target_days: int):
        d = chain.copy()
        d["dte_diff"] = (d["dte"] - target_days).abs()
        if d.empty:
            return None
        exp_used = d.sort_values("dte_diff").iloc[0]["expiry"]
        e = d[d["expiry"] == exp_used].copy()
        e["k_diff"] = (e["strike"] - spot).abs()
        atm_k = float(e.sort_values("k_diff").iloc[0]["strike"])
        atm = e[e["strike"] == atm_k]
        ivs = atm["impliedVolatility"].dropna().tolist()
        ivs = [float(x) for x in ivs if x and x > 0]
        return float(np.mean(ivs)) if ivs else None

    for bucket_name, days in [("Short (≤14D)", SHORT_DAYS), ("Mid (15–180D)", MID_DAYS), ("Long (≥181D)", LONG_DAYS)]:
        sub = chain[chain["bucket"] == bucket_name].copy()
        puts = sub[sub["option_type"] == "put"]
        calls = sub[sub["option_type"] == "call"]

        sup = puts[puts["strike"] <= spot].groupby("strike", as_index=False)[strength_metric].sum().sort_values(strength_metric, ascending=False).head(3)
        res = calls[calls["strike"] >= spot].groupby("strike", as_index=False)[strength_metric].sum().sort_values(strength_metric, ascending=False).head(3)

        sigma = pick_atm_iv_from_loaded(days)
        sigma = sigma if (sigma and sigma > 0) else hv_fallback
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

        sup = add_probs(sup, "support")
        res = add_probs(res, "resistance")

        st.markdown(f"### {bucket_name}")
        st.caption(f"σ used: {sigma if sigma else 'NA'} | q={q:.3f}")
        cL, cR = st.columns(2)
        with cL:
            st.markdown("**Support**")
            st.dataframe(sup, use_container_width=True, height=160)
        with cR:
            st.markdown("**Resistance**")
            st.dataframe(res, use_container_width=True, height=160)

with tabs[3]:
    st.subheader("Gamma Wall (Proxy)")
    d = chain.copy()
    d = d[d["impliedVolatility"].notna() & (d["impliedVolatility"] > 0)].copy()
    if d.empty:
        st.warning("No IV available in loaded chains for gamma computation.")
    else:
        d["T"] = d["dte"] / 365.0
        d["gamma"] = [
            (bs_gamma(spot, float(r["strike"]), float(r["T"]), r=0.0, q=q, sigma=float(r["impliedVolatility"])) or np.nan)
            for _, r in d.iterrows()
        ]
        d["gex_proxy"] = d["gamma"] * d["openInterest"] * CONTRACT_MULTIPLIER * (spot ** 2)
        agg = d.groupby("strike", as_index=False)["gex_proxy"].sum().sort_values("strike")
        fig = px.bar(agg, x="strike", y="gex_proxy")
        fig.update_traces(marker_color="#FF991C")
        ymax = max(1.0, float(np.nanmax(agg["gex_proxy"].values)))
        fig = add_spot_line(fig, spot, ymax)
        fig = style_fig(fig, "Gamma Exposure Proxy by Strike")
        st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    st.subheader("IV Surface (Heatmap)")
    d = chain[chain["impliedVolatility"].notna() & (chain["impliedVolatility"] > 0)].copy()
    if d.empty:
        st.warning("No IV data in loaded chains.")
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
        figH.update_layout(template="plotly_dark", height=560, title="IV Heatmap: Expiry × Strike",
                           paper_bgcolor="#000000", plot_bgcolor="#000000", font=dict(color="#E6E6E6"))
        st.plotly_chart(figH, use_container_width=True)

with tabs[5]:
    st.subheader("Report Builder")
    sections = st.multiselect(
        "Include sections",
        [
            "Header",
            "Aggregated OI (Top 15)",
            "Aggregated Volume (Top 15)",
            f"Selected Expiry ({selected_exp}) OI (Top 15)",
            "S/R + Probabilities"
        ],
        default=["Header", "Aggregated OI (Top 15)", "S/R + Probabilities"]
    )

    def md_table(df: pd.DataFrame, max_rows=20):
        if df is None or df.empty:
            return "_No data._"
        return df.head(max_rows).to_markdown(index=False)

    def build_report():
        lines = []
        lines.append(f"# {APP_TITLE}")
        lines.append(f"**Ticker:** {ticker}")
        lines.append(f"**Generated (UTC):** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Mode:** {mode}")
        lines.append(f"**Loaded expiries this run:** {len(expiries_to_load)} / {len(expiries)}")
        lines.append("")

        if "Header" in sections:
            lines.append("## Snapshot")
            lines.append(f"- Spot: **{spot:,.2f}**")
            lines.append(f"- HV(21): **{(hv21*100):.2f}%**" if hv21 else "- HV(21): NA")
            lines.append(f"- HV(63): **{(hv63*100):.2f}%**" if hv63 else "- HV(63): NA")
            lines.append("")

        if "Aggregated OI (Top 15)" in sections:
            df = strike_3bar_frame(chain, "openInterest").sort_values("Total", ascending=False).head(15)
            lines.append("## Aggregated Open Interest (Top 15)")
            lines.append(md_table(df, 15))
            lines.append("")

        if "Aggregated Volume (Top 15)" in sections:
            df = strike_3bar_frame(chain, "volume").sort_values("Total", ascending=False).head(15)
            lines.append("## Aggregated Volume (Top 15)")
            lines.append(md_table(df, 15))
            lines.append("")

        if f"Selected Expiry ({selected_exp}) OI (Top 15)" in sections:
            d = chain[chain["expiry"] == selected_exp]
            df = strike_3bar_frame(d, "openInterest").sort_values("Total", ascending=False).head(15)
            lines.append(f"## Selected Expiry OI (Top 15) — {selected_exp}")
            lines.append(md_table(df, 15))
            lines.append("")

        if "S/R + Probabilities" in sections:
            lines.append("## Support/Resistance + Probabilities")
            lines.append("_Computed from loaded expiries only in Lean mode._")
            lines.append("")

        return "\n".join(lines)

    rep = build_report()
    st.text_area("Preview (Markdown)", rep, height=380)
    st.download_button(
        "Download .md",
        data=rep.encode("utf-8"),
        file_name=f"{ticker}_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}Z.md",
        mime="text/markdown"
    )

st.markdown("---")
st.caption("If you want true full-universe aggregation on Cloud with no rate limits, you must precompute outside the app (scheduled job) and serve cached results. Yahoo will not reliably support full expiry sweeps from shared IPs.")
