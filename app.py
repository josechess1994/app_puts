import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import concurrent.futures
from itertools import repeat
from datetime import datetime, timedelta
from scipy.stats import norm
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from functools import lru_cache

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

@lru_cache(maxsize=1024)
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

@lru_cache(maxsize=1024)
def get_daily_closes(symbol, lookback_days=400):
    try:
        end = datetime.now().date()
        start = end - timedelta(days=lookback_days)
        r = SESSION.get(
            f"{BASE_URL}/markets/history",
            params={
                "symbol": symbol,
                "interval": "daily",
                "start": start.strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%d"),
            },
            timeout=10,
        )
        history = r.json().get("history", {}).get("day", [])
        if isinstance(history, dict):
            history = [history]
        closes = [float(d.get("close")) for d in history if d.get("close") is not None]
        return closes
    except Exception:
        return []


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

def calcular_metricas_desde_closes(price, closes, include_trend=True):
    try:
        if price is None or not closes:
            return None, None, None, None, None, None, None, None

        ser = pd.Series(closes, dtype="float64")
        if ser.empty:
            return None, None, None, None, None, None, None, None

        # Cambios aproximados por días hábiles
        def _pct_from_lookback(lookback):
            if len(ser) <= lookback:
                return None
            inicio = float(ser.iloc[-(lookback + 1)])
            fin = float(ser.iloc[-1])
            if inicio == 0:
                return None
            return round((fin - inicio) / inicio * 100, 2)

        cambio_1m = _pct_from_lookback(21)
        cambio_2m = _pct_from_lookback(42)
        cambio_3m = _pct_from_lookback(63)
        cambio_6m = _pct_from_lookback(126)

        ivr = None
        if len(ser) >= 252:
            ivs = ser.pct_change().rolling(21).std() * np.sqrt(252)
            ivs = ivs.dropna()
            if not ivs.empty:
                act = ivs.iloc[-1]
                denom = float(ivs.max() - ivs.min()) or 1e-8
                ivr = round((act - ivs.min()) / denom * 100, 2)

        trend_status, pct_from_ma50, pct_from_ma200 = calcular_trend_status(price, closes)
        return ivr, cambio_1m, cambio_2m, cambio_3m, pct_from_ma50, pct_from_ma200, trend_status
    except Exception:
        return None, None, None, None, None, None, None, None

def _fetch_earnings_pair(ticker):
    return ticker, obtener_proximo_earnings(ticker)


def prefetch_earnings_map(tickers, max_workers=8):
    if not tickers:
        return {}
    workers = max(1, min(int(max_workers or 1), len(tickers), 32))
    out = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for ticker, dt in ex.map(_fetch_earnings_pair, tickers):
            out[ticker] = dt
    return out


def _pop_from_delta(delta_val):
    try:
        return round((1 - abs(float(delta_val))) * 100, 1)
    except Exception:
        return None

# =========================
# CONSTRUCCIÓN DE BASE
# =========================
def procesar_ticker(
    ticker,
    option_types=("put", "call"),
    dias_range=(1, 60),
    max_expirations=0,
    atm_window_range=(0, 100),
    include_earnings=False,
    include_trend=True,
    earnings_map=None,
):
    registros = []
    q = get_quote(ticker)
    last = q.get("last")
    if last is None:
        return registros

    closes = get_daily_closes(ticker)
    ivr, cambio_1m, cambio_2m, cambio_3m, cambio_6m, pct_from_ma50, pct_from_ma200, trend_status = calcular_metricas_desde_closes(last, closes, include_trend=include_trend)
    prox_earnings = (earnings_map or {}).get(ticker) if include_earnings else None

    expirations = get_expirations(ticker)
    valid = []
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

def procesar_ticker_safe(ticker, option_types, dias_range, max_expirations, atm_window_range, include_earnings, include_trend, earnings_map):
    try:
        return procesar_ticker(ticker, option_types, dias_range, max_expirations, atm_window_range, include_earnings, include_trend, earnings_map)
    except Exception:
        # Si algo raro pasa en un ticker, seguimos con los demás
        return []

def cargar_base(
    tickers,
    option_types,
    dias_range,
    max_expirations,
    atm_window_range,
    include_earnings=False,
    include_trend=True,
    max_workers=20,
    earnings_workers=8,
):
    all_regs = []
    earnings_map = prefetch_earnings_map(tickers, max_workers=earnings_workers) if include_earnings else {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for regs in executor.map(
            procesar_ticker_safe,
            tickers,
            repeat(tuple(option_types)),
            repeat(tuple(dias_range)),
            repeat(max_expirations),
            repeat(tuple(atm_window_range)),
            repeat(include_earnings),
            repeat(include_trend),
            repeat(earnings_map),
        ):
            all_regs.extend(regs)
    return pd.DataFrame(all_regs)

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

# Preset
st.sidebar.header("Preset")
if st.sidebar.button("Preset (20–45 DTE, Δ −0.30 a +0.30, IV ≥ 25%, IVR ≥ 30%)"):
    st.session_state["k_dias"] = (20, 45)
    st.session_state["k_delta"] = (-0.35, -0.05)
    st.session_state["k_iv"] = (25.0, 100.0)
    st.session_state["k_ivr"] = (30.0, 100.0)
    st.session_state["k_ch"] = (-100.0, 5.0)
    st.session_state["k_ch_2m"] = (-100.0, 10.0)
    st.session_state["k_ch_3m"] = (-100.0, 100.0)
    st.session_state["k_ch_6m"] = (-100.0, 100.0)
    st.rerun()

# 1) Configurar Base
st.sidebar.header("1. Configurar Base")
r_dias = st.sidebar.slider("Días hasta expiración", 1, 60, st.session_state.get("k_dias", (1, 60)), key="k_dias")
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
use_earnings_filter = st.sidebar.checkbox("Usar filtro Earnings", value=st.session_state.get("k_use_earnings_filter", True), key="k_use_earnings_filter")
r_earnings = st.sidebar.selectbox("Earnings antes de expiración", ["Todos", "Solo con earnings", "Solo sin earnings"], key="k_earnings")
include_earnings = st.sidebar.checkbox("Calcular earnings (más lento)", value=st.session_state.get("k_include_earnings", False), key="k_include_earnings")
earnings_workers = st.sidebar.number_input("Hilos earnings", 1, 32, 8, key="k_earnings_workers") if include_earnings else 8
use_trend_filter = st.sidebar.checkbox("Usar filtro Trend Status", value=st.session_state.get("k_use_trend_filter", True), key="k_use_trend_filter")
include_trend = use_trend_filter
r_trend = st.sidebar.multiselect(
    "Trend Status",
    ["STRONG_UPTREND", "PULLBACK", "TRANSITION", "DOWNTREND"],
    default=st.session_state.get("k_trend", ["STRONG_UPTREND", "PULLBACK", "TRANSITION", "DOWNTREND"]),
    key="k_trend",
)
workers = st.sidebar.number_input("Hilos (workers)", 2, 50, 20, key="k_workers")

if st.sidebar.button("🔄 Cargar base") and option_types_to_load and selected_tickers:
    with st.spinner(
        f"Cargando {len(selected_tickers)} tickers, tipos {option_types_to_load}, días {r_dias}, ATM {atm_win}…"
    ):
        df_base = cargar_base(
            selected_tickers,
            option_types_to_load,
            r_dias,
            max_exp,
            atm_win,
            include_earnings=include_earnings,
            include_trend=include_trend,
            max_workers=workers,
            earnings_workers=earnings_workers,
        )
    st.session_state["base_df"] = df_base
    st.success(f"Base cargada: {len(df_base)} contratos")

# 2) Mostrar/filtrar base
if "base_df" in st.session_state and not st.session_state["base_df"].empty:
    base = st.session_state["base_df"].copy()

    # Compatibilidad hacia atrás para bases cargadas antes de nuevos campos
    defaults = {
        "Cambio 6M (%)": None,
        "Próximo Earnings": None,
        "Días a Earnings": None,
        "Earnings antes exp": "No",
        "Trend Status": None,
        "Pct from MA50 (%)": None,
        "Pct from MA200 (%)": None,
    }
    for col, val in defaults.items():
        if col not in base.columns:
            base[col] = val

    tipos_vista = sorted(base["OptionType"].dropna().unique().tolist())
    tipos_vista_sel = st.sidebar.multiselect(
        "Tipos a mostrar", ["put", "call"], default=tipos_vista if tipos_vista else ["put", "call"], key="k_types_view"
    )

    mask_earnings = pd.Series(True, index=base.index)
    if use_earnings_filter:
        if r_earnings != "Todos" and not include_earnings:
            st.sidebar.warning("Activa 'Calcular earnings (más lento)' para filtrar por earnings.")
        elif r_earnings == "Solo con earnings":
            mask_earnings = base["Earnings antes exp"] == "Sí"
        elif r_earnings == "Solo sin earnings":
            mask_earnings = base["Earnings antes exp"] == "No"

    mask_trend = pd.Series(True, index=base.index)
    if use_trend_filter and r_trend:
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
    st.dataframe(df, use_container_width=True)

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
        st.dataframe(out, use_container_width=True)
    else:
        st.caption("Ajusta parámetros y pulsa 'Aplicar estrategia' para ver resultados.")
else:
    st.info("Carga la base desde la barra lateral.")
