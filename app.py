
import os
import json
import time
import streamlit as st
import pandas as pd
import numpy as np
import requests
import concurrent.futures
from itertools import repeat
from datetime import datetime, timedelta
from datetime import datetime
from scipy.stats import norm
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from functools import lru_cache

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache", "daily_closes")
DAILY_CLOSES_TTL_SECONDS = 24 * 60 * 60
HISTORY_LOOKBACK_DAYS = 260
TREND_HISTORY_TIMEOUT = 6


def _daily_cache_path(symbol):
    safe_symbol = (symbol or "").replace("/", "_").replace("\\", "_")
    return os.path.join(CACHE_DIR, f"{safe_symbol}.json")


def _read_daily_cache(symbol, now_ts):
    path = _daily_cache_path(symbol)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        ts = float(payload.get("ts", 0))
        closes = payload.get("closes", [])
        if (now_ts - ts) <= DAILY_CLOSES_TTL_SECONDS and isinstance(closes, list):
            return closes
    except Exception:
        return None
    return None


def _write_daily_cache(symbol, closes, now_ts):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(_daily_cache_path(symbol), "w", encoding="utf-8") as fh:
            json.dump({"ts": now_ts, "closes": closes}, fh)
    except Exception:
        return

# =========================
# TOKEN SEGURO (secrets / env)
# =========================
def _get_tradier_token():
    tok = None
    try:
        tok = st.secrets.get("TRADIER_TOKEN", None)
    except Exception:
        pass
    if not tok:
        tok = os.getenv("TRADIER_TOKEN")
    return tok

BASE_URL = "https://api.tradier.com/v1"
TOKEN = "aLvdjAoMpkuiGLVzC1hgaAAGIv9I"
if not TOKEN:
    st.set_page_config(page_title="Filtro Estrategias Opciones", layout="wide")
    st.error("Falta el token. Añade TRADIER_TOKEN en Settings→Secrets o como variable de entorno.")
    st.stop()
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/json"}

# Sesión HTTP con reintentos
SESSION = requests.Session()
SESSION.headers.update(HEADERS)
_adapter = HTTPAdapter(
    pool_connections=100,
    pool_maxsize=100,
    max_retries=Retry(total=2, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504]),
)
SESSION.mount("https://", _adapter)
SESSION.mount("http://", _adapter)

# =========================
# AUXILIARES DE DATOS
# =========================
@st.cache_data(ttl=86400)
def obtener_tickers_sp500():
    import io

    def _normalize_syms(syms):
        return [s.replace(".", "-").strip().upper() for s in syms if isinstance(s, str) and s.strip()]

    # 1) GitHub dataset
    try:
        url_csv = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        r = requests.get(url_csv, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        syms = _normalize_syms(df["Symbol"].tolist())
        if syms:
            return syms
    except Exception:
        pass

    # 2) Wikipedia
    try:
        url_wiki = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        r = requests.get(url_wiki, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        tablas = pd.read_html(r.text)
        syms = _normalize_syms(tablas[0]["Symbol"].tolist())
        if syms:
            return syms
    except Exception:
        pass

    # 3) Fallback mínimo
    return _normalize_syms([
        "AAPL","MSFT","AMZN","NVDA","GOOGL","META","BRK.B","JPM","UNH","XOM",
        "V","HD","LLY","PG","MA","AVGO","COST","JNJ","WMT","ORCL"
    ])

# =========================
# CÁLCULOS (Greeks, IV, etc.)
# =========================
def calcular_delta(S, K, T, r, sigma):
    if sigma is None or sigma <= 0 or T <= 0:
        return None
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1  # delta de put

def black_scholes_put_iv(S, K, T, price=0.01, r=0.04):
    eps, sigma = 1e-5, 0.3
    for _ in range(100):
        if sigma <= 0:
            sigma = 1e-3
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        model = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        vega = max(S * np.sqrt(T) * norm.pdf(d1), 1e-8)
        diff = model - price
        if abs(diff) < eps:
            return sigma
        sigma = max(sigma - diff / vega, 1e-3)
    return sigma

@lru_cache(maxsize=1024)
def calcular_iv_rank(ticker):
    try:
        closes = yf.Ticker(ticker).history(period="1y")["Close"]
        if closes is None or closes.empty:
            return None
        ivs = closes.pct_change().rolling(21).std() * np.sqrt(252)
        ivs = ivs.dropna()
        if ivs.empty:
            return None
        act = ivs.iloc[-1]
        denom = float(ivs.max() - ivs.min()) or 1e-8
        return round((act - ivs.min()) / denom * 100, 2)
    except Exception:
        return None

@lru_cache(maxsize=2048)
def obtener_cambio_periodo(ticker, period="1mo"):
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist is None or hist.empty or "Close" not in hist:
            return None
        close = hist["Close"]
        if len(close) < 2:
            return None
        inicio, fin = float(close.iloc[0]), float(close.iloc[-1])
        if inicio == 0:
            return None
        return round((fin - inicio) / inicio * 100, 2)
    except Exception:
        return None

@st.cache_data(ttl=86400, show_spinner=False)
def obtener_proximo_earnings(ticker):
    try:
        ed = yf.Ticker(ticker).get_earnings_dates(limit=4)
        if ed is None or ed.empty:
            return None
        now_naive = datetime.now()
        for idx in ed.index:
            dt = pd.to_datetime(idx)
            if pd.isna(dt):
                continue
            dt = dt.to_pydatetime().replace(tzinfo=None)
            if dt >= now_naive:
                return dt
        return None
    except Exception:
        return None


# =========================
# API TRADIER (helpers)
# =========================
def get_expirations(symbol):
    try:
        r = SESSION.get(f"{BASE_URL}/markets/options/expirations", params={"symbol": symbol}, timeout=10)
        return r.json().get("expirations", {}).get("date", [])
    except Exception:
        return []

def get_option_chain(symbol, expiration):
    try:
        r = SESSION.get(
            f"{BASE_URL}/markets/options/chains",
            params={"symbol": symbol, "expiration": expiration, "greeks": "true"},
            timeout=10,
        )
        return r.json().get("options", {}).get("option", [])
    except Exception:
        return []

def get_quote(symbol):
    try:
        r = SESSION.get(f"{BASE_URL}/markets/quotes", params={"symbols": symbol}, timeout=10)
        data = r.json().get("quotes", {}).get("quote")
        return data[0] if isinstance(data, list) else (data or {})
    except Exception:
        return {}

@st.cache_data(ttl=30, show_spinner=False)
def get_quotes_batch(symbols):
    if not symbols:
        return {}
    try:
        r = SESSION.get(f"{BASE_URL}/markets/quotes", params={"symbols": ",".join(symbols)}, timeout=12)
        data = r.json().get("quotes", {}).get("quote")
        if isinstance(data, dict):
            data = [data]
        return {q.get("symbol"): q for q in (data or []) if isinstance(q, dict) and q.get("symbol")}
    except Exception:
        return {}

def get_daily_closes(symbol, lookback_days=HISTORY_LOOKBACK_DAYS, _today=None):
    now_ts = time.time()
    cached = _read_daily_cache(symbol, now_ts)
    if cached is not None:
        return cached

    try:
        end = _today or datetime.now().date()
        start = end - timedelta(days=lookback_days)
        r = SESSION.get(
            f"{BASE_URL}/markets/history",
            params={
                "symbol": symbol,
                "interval": "daily",
                "start": start.strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%d"),
            },
            timeout=TREND_HISTORY_TIMEOUT,
        )
        history = r.json().get("history", {}).get("day", [])
        if isinstance(history, dict):
            history = [history]
        closes = [float(d.get("close")) for d in history if d.get("close") is not None]
        if closes:
            _write_daily_cache(symbol, closes, now_ts)
        return closes
    except Exception:
        return []


def get_trend_closes_cached(symbol, _today=None):
    return get_daily_closes(symbol, _today=_today)


def calcular_trend_status(price, closes):
    try:
        if price is None or len(closes) < 200:
            return None, None, None

        ser = pd.Series(closes, dtype="float64")
        sma50 = ser.rolling(50).mean().iloc[-1]
        sma200 = ser.rolling(200).mean().iloc[-1]
        if pd.isna(sma50) or pd.isna(sma200) or sma50 == 0 or sma200 == 0:
            return None, None, None

        pct_from_ma50 = (price - sma50) / sma50
        pct_from_ma200 = (price - sma200) / sma200

        if price > sma50 and sma50 > sma200:
            trend = "STRONG_UPTREND"
        elif price < sma50 and price > sma200 and sma50 > sma200:
            trend = "PULLBACK"
        elif price > sma200 and sma50 <= sma200:
            trend = "TRANSITION"
        elif price < sma200 and sma50 < sma200:
            trend = "DOWNTREND"
        else:
            trend = "TRANSITION"

        return trend, round(pct_from_ma50 * 100, 2), round(pct_from_ma200 * 100, 2)
    except Exception:
        return None, None, None

def _pop_from_delta(delta_val):
    try:
        return round((1 - abs(float(delta_val))) * 100, 1)
    except Exception:
        return None


def _compute_trend_for_ticker(symbol, price, today):
    closes = get_trend_closes_cached(symbol, _today=today)
    trend_status, pct_from_ma50, pct_from_ma200 = calcular_trend_status(price, closes)
    return symbol, trend_status, pct_from_ma50, pct_from_ma200


TREND_BADGE = {
    "STRONG_UPTREND": "🟢 STRONG_UPTREND",
    "PULLBACK": "🟡 PULLBACK",
    "TRANSITION": "🟠 TRANSITION",
    "DOWNTREND": "🔴 DOWNTREND",
}


def _earnings_pill(days_to_earnings):
    if pd.isna(days_to_earnings):
        return "⚪ Unknown"
    days = int(days_to_earnings)
    if days > 21:
        return f"🟢 Earnings in {days}d"
    if days >= 7:
        return f"🟡 Earnings in {days}d"
    return f"🔴 Earnings in {days}d"


def _render_active_filters_summary(r_dias, r_dlt, r_iv, r_ir, sel_horizon, r_earnings, r_trend, option_types_to_load):
    chips = [
        f"DTE: {r_dias[0]}–{r_dias[1]}",
        f"Delta: {r_dlt[0]:.2f} to {r_dlt[1]:.2f}",
        f"IV min: {r_iv[0]:.1f}%",
        f"IVR min: {r_ir[0]:.1f}",
        f"Horizon: {sel_horizon}",
        f"Earnings: {r_earnings}",
        f"Trend: {'ON' if r_trend else 'OFF'}",
        f"Ops: {', '.join(option_types_to_load)}",
    ]
    st.markdown("### 🎛️ Active Filters Summary")
    st.caption(" • ".join(chips))


def _render_formatted_table(df, cols_to_show):
    to_show = df[cols_to_show].copy()

    if "Trend Status" in to_show.columns:
        to_show["Trend Badge"] = to_show["Trend Status"].map(TREND_BADGE).fillna("⚪ Pending")
    if "Días a Earnings" in to_show.columns:
        to_show["Earnings Pill"] = to_show["Días a Earnings"].apply(_earnings_pill)

    if "Ticker" in to_show.columns:
        ordered = ["Ticker"] + [c for c in to_show.columns if c != "Ticker"]
        to_show = to_show[ordered]

    percent_like = [c for c in to_show.columns if "%" in c and pd.api.types.is_numeric_dtype(to_show[c])]
    money_like = [c for c in ["Mid", "Mid Credit", "Mid Credit Total", "Strike", "Short Strike", "Long Strike", "Put Short Strike", "Put Long Strike", "Call Short Strike", "Call Long Strike", "Central Strike"] if c in to_show.columns]

    config = {
        c: st.column_config.NumberColumn(format="%.1f%%") for c in percent_like
    }
    for c in money_like:
        config[c] = st.column_config.NumberColumn(format="$%.2f")
    if "Dias" in to_show.columns:
        config["Dias"] = st.column_config.NumberColumn(format="%d")

    st.dataframe(to_show, use_container_width=True, hide_index=True, column_config=config)

# =========================
# CONSTRUCCIÓN DE BASE
# =========================
def procesar_ticker(
    ticker,
    option_types=("put", "call"),
    dias_range=(1, 60),
    max_expirations=0,
    atm_window_range=(0, 100),
    include_earnings=True,
    include_trend=True,
    quote_data=None,
    valid_expirations=None,
):
    registros = []
    q = quote_data or get_quote(ticker)
    last = q.get("last")
    if last is None:
        return registros

    ivr = calcular_iv_rank(ticker)
    cambio_1m = obtener_cambio_periodo(ticker, "1mo")
    cambio_2m = obtener_cambio_periodo(ticker, "2mo")
    cambio_3m = obtener_cambio_periodo(ticker, "3mo")
    cambio_6m = obtener_cambio_periodo(ticker, "6mo")
    today = datetime.now().date()
    prox_earnings = obtener_proximo_earnings(ticker) if include_earnings else None
    trend_status, pct_from_ma50, pct_from_ma200 = None, None, None

    valid = list(valid_expirations or [])
    if not valid:
        expirations = get_expirations(ticker)
        for exp in expirations:
            try:
                dias = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days
            except Exception:
                continue
            if dias <= 0:
                continue
            if dias_range[0] <= dias <= dias_range[1]:
                valid.append((exp, dias))
    valid.sort(key=lambda x: x[1])
    if max_expirations and max_expirations > 0:
        valid = valid[:max_expirations]

    atm_min, atm_max = atm_window_range if atm_window_range else (0, 100)

    for exp, dias in valid:
        T = dias / 365
        try:
            exp_dt = datetime.strptime(exp, "%Y-%m-%d")
        except Exception:
            exp_dt = None

        earnings_en_ciclo = bool(prox_earnings and exp_dt and datetime.now() <= prox_earnings <= exp_dt)
        dias_a_earnings = (prox_earnings - datetime.now()).days if prox_earnings else None

        chain = get_option_chain(ticker, exp)
        if not chain:
            continue
        for o in chain:
            otype = o.get("option_type")
            if otype not in option_types:
                continue

            strike = o.get("strike")
            if strike is None:
                continue

            # Ventana ATM: |K−S|/S*100 entre [atm_min, atm_max]
            mny = abs(strike - last) / max(last, 1e-8) * 100.0
            if not (atm_min <= mny <= atm_max):
                continue

            bid = o.get("bid") or 0
            ask = o.get("ask") or 0
            mid = (bid + ask) / 2
            if mid <= 0:
                continue

            greeks = o.get("greeks") or {}
            iv_dec = greeks.get("mid_iv") or black_scholes_put_iv(last, strike, T, mid)
            iv_pct = iv_dec * 100 if iv_dec is not None else None
            delta = greeks.get("delta") or calcular_delta(last, strike, T, 0.04, iv_dec)
            ret = (mid / strike) / (max(dias, 1) / 30) * 100

            registros.append(
                {
                    "Ticker": ticker,
                    "Expiración": exp,
                    "OptionType": otype,
                    "Dias": dias,
                    "Strike": strike,
                    "Mid": round(mid, 4),
                    "Retorno %": round(ret, 2),
                    "Delta": round(delta or 0, 3),
                    "POP (%)": _pop_from_delta(delta),
                    "IV (%)": round(iv_pct, 2) if iv_pct is not None else None,
                    "IV Rank": ivr,
                    "Cambio 1M (%)": cambio_1m,
                    "Cambio 2M (%)": cambio_2m,
                    "Cambio 3M (%)": cambio_3m,
                    "Cambio 6M (%)": cambio_6m,
                    "Próximo Earnings": prox_earnings.strftime("%Y-%m-%d") if prox_earnings else None,
                    "Días a Earnings": dias_a_earnings,
                    "Earnings antes exp": "Sí" if earnings_en_ciclo else "No",
                    "Trend Status": trend_status,
                    "Pct from MA50 (%)": pct_from_ma50,
                    "Pct from MA200 (%)": pct_from_ma200,
                }
            )
    return registros

def procesar_ticker_safe(args):
    try:
        return procesar_ticker(*args)
    except Exception:
        return []

def _valid_expirations_for_ticker(ticker, dias_range, max_expirations):
    valid = []
    for exp in get_expirations(ticker):
        try:
            dias = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days
        except Exception:
            continue
        if dias <= 0:
            continue
        if dias_range[0] <= dias <= dias_range[1]:
            valid.append((exp, dias))
    valid.sort(key=lambda x: x[1])
    if max_expirations and max_expirations > 0:
        valid = valid[:max_expirations]
    return ticker, valid

def cargar_base(
    tickers,
    option_types,
    dias_range,
    max_expirations,
    atm_window_range,
    max_workers=20,
    include_earnings=True,
    include_trend=True,
    quick_trend=False,
    quick_trend_n=120,
    progress_callback=None,
):
    all_regs = []
    stage_stats = {"phase_a_tickers": len(tickers), "phase_b_tickers": 0, "trend_tickers": 0}

    if progress_callback:
        progress_callback("Fetching quotes", 0.25)

    quotes_map = {}
    for i in range(0, len(tickers), 200):
        chunk = tickers[i:i + 200]
        quotes_map.update(get_quotes_batch(chunk))

    if progress_callback:
        progress_callback("Applying base filters", 0.45)

    tickers_with_quotes = [t for t in tickers if (quotes_map.get(t, {}) or {}).get("last") is not None]
    exp_map = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for t, valid in executor.map(_valid_expirations_for_ticker, tickers_with_quotes, repeat(tuple(dias_range)), repeat(max_expirations)):
            if valid:
                exp_map[t] = valid

    phase_b_tickers = list(exp_map.keys())
    stage_stats["phase_b_tickers"] = len(phase_b_tickers)

    if progress_callback:
        progress_callback("Fetching options chains", 0.65)
    if include_earnings and progress_callback:
        progress_callback("Computing earnings", 0.78)
    if include_trend and progress_callback:
        progress_callback("Computing trend status", 0.88)

    ticker_args = [
        (
            t,
            tuple(option_types),
            tuple(dias_range),
            max_expirations,
            tuple(atm_window_range),
            include_earnings,
            include_trend,
            quotes_map.get(t),
            exp_map.get(t),
        )
        for t in phase_b_tickers
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for regs in executor.map(procesar_ticker_safe, ticker_args):
            all_regs.extend(regs)

    if include_trend and all_regs:
        today = datetime.now().date()
        trend_targets = {
            reg["Ticker"]: (quotes_map.get(reg["Ticker"], {}) or {}).get("last")
            for reg in all_regs
        }
        trend_targets = {t: p for t, p in trend_targets.items() if p is not None}

        if quick_trend and quick_trend_n > 0 and trend_targets:
            ranked = sorted(
                all_regs,
                key=lambda r: float(r.get("Retorno %", float("-inf"))),
                reverse=True,
            )
            top_symbols = []
            seen = set()
            for reg in ranked:
                sym = reg.get("Ticker")
                if sym in trend_targets and sym not in seen:
                    top_symbols.append(sym)
                    seen.add(sym)
                if len(top_symbols) >= quick_trend_n:
                    break
            trend_targets = {s: trend_targets[s] for s in top_symbols}

        stage_stats["trend_tickers"] = len(trend_targets)
        if trend_targets:
            trend_map = {}
            trend_workers = max(2, min(max_workers, 30))
            with concurrent.futures.ThreadPoolExecutor(max_workers=trend_workers) as executor:
                for symbol, trend_status, pct_from_ma50, pct_from_ma200 in executor.map(
                    _compute_trend_for_ticker,
                    trend_targets.keys(),
                    trend_targets.values(),
                    repeat(today),
                ):
                    trend_map[symbol] = (trend_status, pct_from_ma50, pct_from_ma200)

            for reg in all_regs:
                trend_vals = trend_map.get(reg["Ticker"])
                if trend_vals:
                    reg["Trend Status"], reg["Pct from MA50 (%)"], reg["Pct from MA200 (%)"] = trend_vals

    return pd.DataFrame(all_regs), stage_stats

# =========================
# ESTRATEGIAS (igual que antes)
# =========================
def put_credit_spread(df, width_range, delta_range, credit_range):
    res = []
    puts = df[df["OptionType"] == "put"]
    for (sym, exp), grp in puts.groupby(["Ticker", "Expiración"]):
        shorts = grp[grp["Delta"].between(delta_range[0], delta_range[1])]
        for _, s in shorts.iterrows():
            for _, l in grp.iterrows():
                w = s.Strike - l.Strike
                mc = s.Mid - l.Mid
                if not (width_range[0] <= w <= width_range[1] and credit_range[0] <= mc <= credit_range[1]):
                    continue
                ret = (mc / max(w, 1)) / (s.Dias / 30) * 100
                res.append(
                    {
                        "Ticker": sym,
                        "Expiración": exp,
                        "Dias": s.Dias,
                        "Short Strike": s.Strike,
                        "Long Strike": l.Strike,
                        "Delta": s.Delta,
                        "POP (%)": _pop_from_delta(s.Delta),
                        "IV (%)": s["IV (%)"],
                        "IV Rank": s["IV Rank"],
                        "Cambio 1M (%)": s["Cambio 1M (%)"],
                        "Cambio 2M (%)": s["Cambio 2M (%)"],
                        "Cambio 3M (%)": s["Cambio 3M (%)"],
                        "Cambio 6M (%)": s["Cambio 6M (%)"],
                        "Próximo Earnings": s["Próximo Earnings"],
                        "Días a Earnings": s["Días a Earnings"],
                        "Earnings antes exp": s["Earnings antes exp"],
                        "Trend Status": s["Trend Status"],
                        "Pct from MA50 (%)": s["Pct from MA50 (%)"],
                        "Pct from MA200 (%)": s["Pct from MA200 (%)"],
                        "Width": w,
                        "Mid Credit": round(mc, 2),
                        "Return %": round(ret, 2),
                    }
                )
    return pd.DataFrame(res)

def bear_call_spread(df, width_range, delta_range, credit_range):
    res = []
    calls = df[df["OptionType"] == "call"]
    for (sym, exp), grp in calls.groupby(["Ticker", "Expiración"]):
        shorts = grp[grp["Delta"].between(delta_range[0], delta_range[1])]
        for _, s in shorts.iterrows():
            for _, l in grp.iterrows():
                w = l.Strike - s.Strike
                mc = s.Mid - l.Mid
                if not (width_range[0] <= w <= width_range[1] and credit_range[0] <= mc <= credit_range[1]):
                    continue
                ret = (mc / max(w, 1)) / (s.Dias / 30) * 100
                res.append(
                    {
                        "Ticker": sym,
                        "Expiración": exp,
                        "Dias": s.Dias,
                        "Short Strike": s.Strike,
                        "Long Strike": l.Strike,
                        "Delta": s.Delta,
                        "POP (%)": _pop_from_delta(s.Delta),
                        "IV (%)": s["IV (%)"],
                        "IV Rank": s["IV Rank"],
                        "Cambio 1M (%)": s["Cambio 1M (%)"],
                        "Cambio 2M (%)": s["Cambio 2M (%)"],
                        "Cambio 3M (%)": s["Cambio 3M (%)"],
                        "Cambio 6M (%)": s["Cambio 6M (%)"],
                        "Próximo Earnings": s["Próximo Earnings"],
                        "Días a Earnings": s["Días a Earnings"],
                        "Earnings antes exp": s["Earnings antes exp"],
                        "Trend Status": s["Trend Status"],
                        "Pct from MA50 (%)": s["Pct from MA50 (%)"],
                        "Pct from MA200 (%)": s["Pct from MA200 (%)"],
                        "Width": w,
                        "Mid Credit": round(mc, 2),
                        "Return %": round(ret, 2),
                    }
                )
    return pd.DataFrame(res)

def iron_condor(df, w_put_range, d_put_range, w_call_range, d_call_range, credit_range):
    pc = put_credit_spread(df, w_put_range, d_put_range, (credit_range[0], float("inf")))
    bc = bear_call_spread(df, w_call_range, d_call_range, (credit_range[0], float("inf")))
    res = []
    for _, p in pc.iterrows():
        matches = bc[(bc.Ticker == p.Ticker) & (bc.Expiración == p.Expiración)]
        for _, c in matches.iterrows():
            tot = p["Mid Credit"] + c["Mid Credit"]
            if not (credit_range[0] <= tot <= credit_range[1]):
                continue
            ret = (tot / (p.Width + c.Width)) / (p.Dias / 30) * 100
            res.append(
                {
                    "Ticker": p.Ticker,
                    "Expiración": p.Expiración,
                    "Dias": p.Dias,
                    "Put Short Strike": p["Short Strike"],
                    "Put Long Strike": p["Long Strike"],
                    "Call Short Strike": c["Short Strike"],
                    "Call Long Strike": c["Long Strike"],
                    "Delta Put": p.Delta,
                    "POP Put (%)": _pop_from_delta(p.Delta),
                    "Delta Call": c.Delta,
                    "POP Call (%)": _pop_from_delta(c.Delta),
                    "IV (%)": p["IV (%)"],
                    "IV Rank": p["IV Rank"],
                    "Cambio 1M (%)": p["Cambio 1M (%)"],
                    "Cambio 2M (%)": p["Cambio 2M (%)"],
                    "Cambio 3M (%)": p["Cambio 3M (%)"],
                    "Cambio 6M (%)": p["Cambio 6M (%)"],
                    "Próximo Earnings": p["Próximo Earnings"],
                    "Días a Earnings": p["Días a Earnings"],
                    "Earnings antes exp": p["Earnings antes exp"],
                    "Trend Status": p["Trend Status"],
                    "Pct from MA50 (%)": p["Pct from MA50 (%)"],
                    "Pct from MA200 (%)": p["Pct from MA200 (%)"],
                    "Width Put": p.Width,
                    "Width Call": c.Width,
                    "Mid Credit Total": round(tot, 2),
                    "Return %": round(ret, 2),
                }
            )
    return pd.DataFrame(res)

def iron_fly(df, width_range, delta_range, credit_range):
    res = []
    calls = df[df["OptionType"] == "call"]
    for (sym, exp), grp in calls.groupby(["Ticker", "Expiración"]):
        shorts = grp[grp["Delta"].between(delta_range[0], delta_range[1])]
        for _, s in shorts.iterrows():
            for w in range(width_range[0], width_range[1] + 1):
                low = grp[grp.Strike == s.Strike - w]
                high = grp[grp.Strike == s.Strike + w]
                if low.empty or high.empty:
                    continue
                mc = 2 * s.Mid - low.Mid.iloc[0] - high.Mid.iloc[0]
                if not (credit_range[0] <= mc <= credit_range[1]):
                    continue
                ret = (mc / (2 * max(w, 1))) / (s.Dias / 30) * 100
                res.append(
                    {
                        "Ticker": sym,
                        "Expiración": exp,
                        "Dias": s.Dias,
                        "Central Strike": s.Strike,
                        "Width": w,
                        "Delta": s.Delta,
                        "POP (%)": _pop_from_delta(s.Delta),
                        "IV (%)": s["IV (%)"],
                        "IV Rank": s["IV Rank"],
                        "Cambio 1M (%)": s["Cambio 1M (%)"],
                        "Cambio 2M (%)": s["Cambio 2M (%)"],
                        "Cambio 3M (%)": s["Cambio 3M (%)"],
                        "Cambio 6M (%)": s["Cambio 6M (%)"],
                        "Próximo Earnings": s["Próximo Earnings"],
                        "Días a Earnings": s["Días a Earnings"],
                        "Earnings antes exp": s["Earnings antes exp"],
                        "Mid Credit": round(mc, 2),
                        "Return %": round(ret, 2),
                    }
                )
    return pd.DataFrame(res)

def jade_lizard(df, w_call_range, d_put_range, d_call_range, credit_range):
    res = []
    for (sym, exp), grp in df.groupby(["Ticker", "Expiración"]):
        sp = grp[(grp.OptionType == "put") & (grp.Delta.between(d_put_range[0], d_put_range[1]))]
        sc = grp[(grp.OptionType == "call") & (grp.Delta.between(d_call_range[0], d_call_range[1]))]
        for _, p in sp.iterrows():
            for _, c in sc.iterrows():
                for w in range(w_call_range[0], w_call_range[1] + 1):
                    lc_strike = c.Strike + w
                    long_call = grp[(grp.OptionType == "call") & (grp.Strike == lc_strike)]
                    if long_call.empty:
                        continue
                    mc = p.Mid + c.Mid - long_call.Mid.iloc[0]
                    if not (credit_range[0] <= mc <= credit_range[1]):
                        continue
                    ret = (mc / max(w, 1)) / (p.Dias / 30) * 100
                    res.append(
                        {
                            "Ticker": sym,
                            "Expiración": exp,
                            "Dias": p.Dias,
                            "Short Put": p.Strike,
                            "Short Call": c.Strike,
                            "Long Call": lc_strike,
                            "Delta Put": p.Delta,
                            "POP Put (%)": _pop_from_delta(p.Delta),
                            "Delta Call": c.Delta,
                            "POP Call (%)": _pop_from_delta(c.Delta),
                            "IV (%)": p["IV (%)"],
                            "IV Rank": p["IV Rank"],
                            "Cambio 1M (%)": p["Cambio 1M (%)"],
                            "Cambio 2M (%)": p["Cambio 2M (%)"],
                            "Cambio 3M (%)": p["Cambio 3M (%)"],
                            "Cambio 6M (%)": p["Cambio 6M (%)"],
                            "Próximo Earnings": p["Próximo Earnings"],
                            "Días a Earnings": p["Días a Earnings"],
                            "Earnings antes exp": p["Earnings antes exp"],
                            "Trend Status": p["Trend Status"],
                            "Pct from MA50 (%)": p["Pct from MA50 (%)"],
                            "Pct from MA200 (%)": p["Pct from MA200 (%)"],
                            "Mid Credit": round(mc, 2),
                            "Return %": round(ret, 2),
                        }
                    )
    return pd.DataFrame(res)

# =========================
# UI STREAMLIT
# =========================
st.set_page_config(page_title="Filtro Estrategias Opciones", layout="wide")
st.title("⚡ Filtro Base + Estrategias (S&P 500)")

# 0) Tickers con “Seleccionar todos / Ninguno”
all_tickers = obtener_tickers_sp500()
if "tickers_sel" not in st.session_state:
    st.session_state["tickers_sel"] = all_tickers[:]  # por defecto todos

colA, colB = st.sidebar.columns(2)
if colA.button("✅ Seleccionar todos"):
    st.session_state["tickers_sel"] = all_tickers[:]
if colB.button("🗑️ Ninguno"):
    st.session_state["tickers_sel"] = []

selected_tickers = st.sidebar.multiselect(
    "Tickers",
    all_tickers,
    default=st.session_state["tickers_sel"],
    key="tickers_sel",
)

# Tipos a incluir al cargar (por defecto SOLO puts)
option_types_to_load = st.sidebar.multiselect(
    "Tipos de opción a incluir al cargar", ["put", "call"], default=["put"], key="k_types_load"
)
if not option_types_to_load:
    st.sidebar.warning("Selecciona al menos un tipo (put/call) para cargar.")

include_earnings = st.sidebar.checkbox("Incluir Earnings", value=True, key="k_include_earnings")
include_trend = st.sidebar.checkbox("Incluir Trend Status (SMA50/SMA200)", value=True, key="k_include_trend")
quick_trend = st.sidebar.toggle("Quick Trend (top N)", value=True, key="k_quick_trend")
quick_trend_n = st.sidebar.number_input("Quick Trend N", 20, 1000, 120, 10, key="k_quick_trend_n")

# Preset
st.sidebar.header("Preset")
if st.sidebar.button("Preset (20–45 DTE, Δ −0.30 a +0.30, IV ≥ 25%, IVR ≥ 30%)"):
    st.session_state["k_dias"] = (20, 45)
    st.session_state["k_delta"] = (-0.30, 0.30)
    st.session_state["k_iv"] = (25.0, 100.0)
    st.session_state["k_ivr"] = (30.0, 100.0)
    st.session_state["k_ch"] = (-100.0, 100.0)
    st.session_state["k_ch_2m"] = (-100.0, 100.0)
    st.session_state["k_ch_3m"] = (-100.0, 100.0)
    st.session_state["k_ch_6m"] = (-100.0, 100.0)
    st.rerun()

horizon_presets = {
    "1 mes": 30,
    "2 meses": 60,
    "3 meses": 90,
    "6 meses": 184,
}
sel_horizon = st.sidebar.selectbox("Horizonte", ["Sin preset"] + list(horizon_presets.keys()), key="k_horizon")
if sel_horizon != "Sin preset":
    st.session_state["k_dias"] = (1, horizon_presets[sel_horizon])
    st.sidebar.caption(f"Rango aplicado: 1-{horizon_presets[sel_horizon]} días")

# 1) Configurar Base
st.sidebar.header("1. Configurar Base")
r_dias = st.sidebar.slider("Días hasta expiración", 1, 184, st.session_state.get("k_dias", (1, 60)), key="k_dias")
max_exp = st.sidebar.number_input("Máx. expiraciones por ticker (0 = sin límite)", 0, 30, 0, 1, key="k_maxexp")
atm_win = st.sidebar.slider("Ventana ATM (min%, max%)", 0, 100, (0, 100), key="k_atm")

# Filtros base (visual)
r_ret = st.sidebar.slider("Retorno mensual (%)", -100.0, 100.0, st.session_state.get("k_ret", (-100.0, 100.0)), key="k_ret")
r_dlt = st.sidebar.slider("Delta", -1.0, 1.0, st.session_state.get("k_delta", (-1.0, 1.0)), key="k_delta")
r_iv = st.sidebar.slider("IV (%)", 0.0, 100.0, st.session_state.get("k_iv", (0.0, 100.0)), key="k_iv")
r_ir = st.sidebar.slider("IV Rank", 0.0, 100.0, st.session_state.get("k_ivr", (0.0, 100.0)), key="k_ivr")
r_ch = st.sidebar.slider("Cambio 1M (%)", -100.0, 100.0, st.session_state.get("k_ch", (-100.0, 100.0)), key="k_ch")
r_ch_2m = st.sidebar.slider("Cambio 2M (%)", -100.0, 100.0, st.session_state.get("k_ch_2m", (-100.0, 100.0)), key="k_ch_2m")
r_ch_3m = st.sidebar.slider("Cambio 3M (%)", -100.0, 100.0, st.session_state.get("k_ch_3m", (-100.0, 100.0)), key="k_ch_3m")
r_ch_6m = st.sidebar.slider("Cambio 6M (%)", -100.0, 100.0, st.session_state.get("k_ch_6m", (-100.0, 100.0)), key="k_ch_6m")
r_earnings = "Todos"
if include_earnings:
    r_earnings = st.sidebar.selectbox("Earnings antes de expiración", ["Todos", "Solo con earnings", "Solo sin earnings"], key="k_earnings")
r_trend = ["STRONG_UPTREND", "PULLBACK", "TRANSITION", "DOWNTREND"]
if include_trend:
    r_trend = st.sidebar.multiselect(
        "Trend Status",
        ["STRONG_UPTREND", "PULLBACK", "TRANSITION", "DOWNTREND"],
        default=st.session_state.get("k_trend", ["STRONG_UPTREND", "PULLBACK", "TRANSITION", "DOWNTREND"]),
        key="k_trend",
    )
workers = st.sidebar.number_input("Hilos (workers)", 2, 50, 30, key="k_workers")

if st.sidebar.button("🔄 Cargar base") and option_types_to_load and selected_tickers:
    progress_bar = st.progress(0)
    status_line = st.empty()
    stage_times = {}
    t0 = time.perf_counter()
    stage_t0 = [t0]

    def mark_stage(stage, pct):
        now = time.perf_counter()
        prev_stage = st.session_state.get("_last_stage_name")
        if prev_stage:
            stage_times[prev_stage] = stage_times.get(prev_stage, 0.0) + (now - stage_t0[0])
        stage_t0[0] = now
        st.session_state["_last_stage_name"] = stage
        progress_bar.progress(int(pct * 100))
        status_line.write(f"⏳ {stage}")

    mark_stage("Loading tickers", 0.1)
    df_base, stage_stats = cargar_base(
        selected_tickers,
        option_types_to_load,
        r_dias,
        max_exp,
        atm_win,
        max_workers=workers,
        include_earnings=include_earnings,
        include_trend=include_trend,
        quick_trend=quick_trend,
        quick_trend_n=int(quick_trend_n),
        progress_callback=mark_stage,
    )
    mark_stage("Building results table", 0.98)

    last_stage = st.session_state.get("_last_stage_name")
    if last_stage:
        stage_times[last_stage] = stage_times.get(last_stage, 0.0) + (time.perf_counter() - stage_t0[0])
    st.session_state.pop("_last_stage_name", None)

    total_time = time.perf_counter() - t0
    progress_bar.progress(100)
    status_line.write("✅ Completado")

    st.session_state["base_df"] = df_base
    st.session_state["base_meta"] = {
        "include_earnings": include_earnings,
        "include_trend": include_trend,
        "stage_times": stage_times,
        "total_time": total_time,
        **stage_stats,
    }
    st.success(f"Base cargada: {len(df_base)} contratos")

# 2) Mostrar/filtrar base
if "base_df" in st.session_state and not st.session_state["base_df"].empty:
    base = st.session_state["base_df"]

    tipos_vista = sorted(base["OptionType"].dropna().unique().tolist())
    tipos_vista_sel = st.sidebar.multiselect(
        "Tipos a mostrar", ["put", "call"], default=tipos_vista if tipos_vista else ["put", "call"], key="k_types_view"
    )

    base_meta = st.session_state.get("base_meta", {})
    base_include_earnings = base_meta.get("include_earnings", True)
    base_include_trend = base_meta.get("include_trend", True)

    mask_earnings = pd.Series(True, index=base.index)
    if base_include_earnings and r_earnings == "Solo con earnings":
        mask_earnings = base["Earnings antes exp"] == "Sí"
    elif base_include_earnings and r_earnings == "Solo sin earnings":
        mask_earnings = base["Earnings antes exp"] == "No"

    mask_trend = pd.Series(True, index=base.index)
    if base_include_trend and r_trend:
        mask_trend = base["Trend Status"].isin(r_trend)

    df = base[
        base["OptionType"].isin(tipos_vista_sel)
        & base["Dias"].between(r_dias[0], r_dias[1])
        & base["Retorno %"].between(r_ret[0], r_ret[1])
        & base["Delta"].between(r_dlt[0], r_dlt[1])
        & base["IV (%)"].between(r_iv[0], r_iv[1])
        & base["IV Rank"].between(r_ir[0], r_ir[1])
        & base["Cambio 1M (%)"].between(r_ch[0], r_ch[1])
        & base["Cambio 2M (%)"].between(r_ch_2m[0], r_ch_2m[1])
        & base["Cambio 3M (%)"].between(r_ch_3m[0], r_ch_3m[1])
        & base["Cambio 6M (%)"].between(r_ch_6m[0], r_ch_6m[1])
        & mask_earnings
        & mask_trend
    ]

    st.subheader(f"🔖 Base filtrada: {len(df)} contratos")
    _render_active_filters_summary(r_dias, r_dlt, r_iv, r_ir, sel_horizon, r_earnings, r_trend, option_types_to_load)

    cols_to_show = df.columns.tolist()
    if not base_include_earnings:
        cols_to_show = [c for c in cols_to_show if c not in ["Próximo Earnings", "Días a Earnings", "Earnings antes exp"]]
    if not base_include_trend:
        cols_to_show = [c for c in cols_to_show if c not in ["Trend Status", "Pct from MA50 (%)", "Pct from MA200 (%)"]]

    df_view = df.copy()
    _render_formatted_table(df_view, cols_to_show)

    if base_meta:
        st.caption(
            f"Tiempo total: {base_meta.get('total_time', 0):.2f}s | "
            f"Fase A tickers: {base_meta.get('phase_a_tickers', 0)} | "
            f"Fase B tickers: {base_meta.get('phase_b_tickers', 0)} | "
            f"Trend tickers: {base_meta.get('trend_tickers', 0)}"
        )
        st.json({k: round(v, 2) for k, v in base_meta.get("stage_times", {}).items()})

    # 3) Estrategias
    st.sidebar.header("2. Estrategias")
    strat = st.sidebar.selectbox(
        "Selecciona estrategia",
        ["Put credit spread", "Bear call spread", "Iron Condor", "Iron Fly", "Jade Lizard", "Broken Wing Butterfly"],
        key="k_strat",
    )

    w_range = st.sidebar.slider("Width range", 1, 20, (1, 5), key="k_width")
    cred_range = st.sidebar.slider("Mid Credit range", -10.0, 10.0, (-1.0, 1.0), key="k_credit")

    out = pd.DataFrame()
    if strat == "Put credit spread":
        d_range = st.sidebar.slider("Delta short put", -1.0, 0.0, (-0.30, -0.15), key="k_dsp")
        if st.sidebar.button("Aplicar estrategia"):
            out = put_credit_spread(df, w_range, d_range, cred_range)

    elif strat == "Bear call spread":
        d_range = st.sidebar.slider("Delta short call", 0.0, 1.0, (0.15, 0.30), key="k_dsc")
        if st.sidebar.button("Aplicar estrategia"):
            out = bear_call_spread(df, w_range, d_range, cred_range)

    elif strat == "Iron Condor":
        dp = st.sidebar.slider("Delta short put", -1.0, 0.0, (-0.30, -0.15), key="k_dsp_ic")
        dc = st.sidebar.slider("Delta short call", 0.0, 1.0, (0.15, 0.30), key="k_dsc_ic")
        if st.sidebar.button("Aplicar estrategia"):
            out = iron_condor(df, w_range, dp, w_range, dc, cred_range)

    elif strat == "Iron Fly":
        d_range = st.sidebar.slider("Delta central", -1.0, 1.0, (-0.15, 0.15), key="k_dfly")
        if st.sidebar.button("Aplicar estrategia"):
            out = iron_fly(df, w_range, d_range, cred_range)

    elif strat == "Jade Lizard":
        dp = st.sidebar.slider("Delta short put", -1.0, 0.0, (-0.30, -0.15), key="k_dsp_j")
        dc = st.sidebar.slider("Delta short call", 0.0, 1.0, (0.15, 0.30), key="k_dsc_j")
        if st.sidebar.button("Aplicar estrategia"):
            out = jade_lizard(df, w_range, dp, dc, cred_range)

    else:  # Broken Wing Butterfly (MVP con iron_fly simétrico)
        d_range = st.sidebar.slider("Delta central", -1.0, 1.0, (-0.15, 0.15), key="k_dbwb")
        if st.sidebar.button("Aplicar estrategia"):
            out = iron_fly(df, w_range, d_range, cred_range)

    if not out.empty:
        st.subheader(f"📊 {strat}: {len(out)} oportunidades")
        _render_formatted_table(out, out.columns.tolist())
    else:
        st.caption("Ajusta parámetros y pulsa 'Aplicar estrategia' para ver resultados.")
else:
    st.info("Carga la base desde la barra lateral.")
