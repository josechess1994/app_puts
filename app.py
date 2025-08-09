import streamlit as st
import pandas as pd
import numpy as np
import requests
import concurrent.futures
from itertools import repeat
from datetime import datetime
from scipy.stats import norm
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- CONFIGURACIÓN ---
BASE_URL = "https://api.tradier.com/v1"
TOKEN    = "aLvdjAoMpkuiGLVzC1hgaAAGIv9I"
HEADERS  = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/json"}

# Pool de conexiones + reintentos para acelerar y ser robustos
SESSION = requests.Session()
SESSION.headers.update(HEADERS)
_adapter = HTTPAdapter(
    pool_connections=100,
    pool_maxsize=100,
    max_retries=Retry(total=2, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504])
)
SESSION.mount("https://", _adapter)
SESSION.mount("http://", _adapter)

# =========================
# AUXILIARES DE DATOS
# =========================
@st.cache_data
def obtener_tickers_sp500():
    tablas = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    return tablas[0]["Symbol"].tolist()

@st.cache_data
def obtener_cambio_mensual(ticker):
    try:
        df = yf.Ticker(ticker).history(period="1mo")["Close"]
        if len(df) >= 2:
            inicio, fin = df.iloc[0], df.iloc[-1]
            return round((fin - inicio) / inicio * 100, 2)
    except:
        pass
    return None

# =========================
# CÁLCULOS (Greeks, IV, etc.)
# =========================
def calcular_delta(S, K, T, r, sigma):
    if sigma is None or sigma == 0 or T == 0:
        return None
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1) - 1  # delta de put

def black_scholes_put_iv(S, K, T, price=0.01, r=0.04):
    # Newton-Raphson simple usando el precio de put europeo
    eps, sigma = 1e-5, 0.3
    for _ in range(100):
        if sigma <= 0:
            sigma = 1e-3
        d1 = (np.log(S/K) + (r+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        model = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        vega  = max(S*np.sqrt(T)*norm.pdf(d1), 1e-8)
        diff  = model - price
        if abs(diff) < eps:
            return sigma
        sigma = sigma - diff/vega
    return max(sigma, 1e-3)

@st.cache_data
def calcular_iv_rank(ticker):
    try:
        closes = yf.Ticker(ticker).history(period="1y")["Close"]
        ivs    = closes.pct_change().rolling(21).std() * np.sqrt(252)
        act    = ivs.iloc[-1]
        return round((act - ivs.min())/(ivs.max() - ivs.min())*100, 2)
    except:
        return None

# =========================
# API TRADIER
# =========================
def get_expirations(symbol):
    try:
        r = SESSION.get(
            f"{BASE_URL}/markets/options/expirations",
            params={"symbol": symbol},
            timeout=10
        )
        return r.json().get("expirations", {}).get("date", [])
    except:
        return []

def get_option_chain(symbol, expiration):
    try:
        r = SESSION.get(
            f"{BASE_URL}/markets/options/chains",
            params={"symbol": symbol, "expiration": expiration, "greeks": "true"},
            timeout=10
        )
        return r.json().get("options", {}).get("option", [])
    except:
        return []

def get_quote(symbol):
    try:
        r = SESSION.get(
            f"{BASE_URL}/markets/quotes",
            params={"symbols": symbol},
            timeout=10
        )
        data = r.json().get("quotes", {}).get("quote")
        return data[0] if isinstance(data, list) else (data or {})
    except:
        return {}

# =========================
# POP helper
# =========================
def _pop_from_delta(delta_val):
    try:
        return round((1 - abs(float(delta_val))) * 100, 1)
    except:
        return None

# =========================
# CONSTRUCCIÓN DE BASE (con filtros tempranos y ventana ATM rango)
# =========================
def procesar_ticker(ticker, option_types, dias_range, max_expirations, atm_window_range):
    registros = []
    quote = get_quote(ticker)
    last  = quote.get("last")
    if last is None:
        return registros

    ivr    = calcular_iv_rank(ticker)
    cambio = obtener_cambio_mensual(ticker)

    expirations = get_expirations(ticker)
    # Filtramos por rango de días y limitamos # de expiraciones si se pide
    valid = []
    for exp in expirations:
        dias = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days
        if dias > 0 and (dias_range[0] <= dias <= dias_range[1]):
            valid.append((exp, dias))
    valid.sort(key=lambda x: x[1])  # por cercanía
    if max_expirations and max_expirations > 0:
        valid = valid[:max_expirations]

    # Ventana ATM como rango [min%, max%]
    atm_min, atm_max = atm_window_range if atm_window_range else (0, 100)

    for exp, dias in valid:
        T = dias / 365
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

            # Ventana ATM: mantener solo si atm_min <= |K−S|/S*100 <= atm_max
            mny = abs(strike - last) / max(last, 1e-8) * 100.0
            if not (atm_min <= mny <= atm_max):
                continue

            bid = o.get("bid") or 0
            ask = o.get("ask") or 0
            mid = (bid + ask) / 2
            if mid <= 0:
                continue

            greeks = o.get("greeks") or {}
            iv     = greeks.get("mid_iv") or black_scholes_put_iv(last, strike, T, mid)
            iv_pct = iv * 100 if iv is not None else None
            delta  = greeks.get("delta") or calcular_delta(last, strike, T, 0.04, iv)
            ret    = (mid / strike) / (dias/30) * 100
            pop    = _pop_from_delta(delta)

            registros.append({
                "Ticker": ticker,
                "OptionType": otype,
                "Expiración": exp,
                "Dias": dias,
                "Strike": strike,
                "Mid": round(mid, 4),
                "Retorno %": round(ret, 2),
                "Delta": round(delta or 0, 3),
                "POP (%)": pop,
                "IV (%)": round(iv_pct, 2) if iv_pct is not None else None,
                "IV Rank": ivr,
                "Cambio 1M (%)": cambio
            })

    return registros

def cargar_base(tickers, option_types, dias_range, max_expirations, atm_window_range, max_workers=20):
    all_regs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for regs in executor.map(
            procesar_ticker,
            tickers,
            repeat(option_types),
            repeat(dias_range),
            repeat(max_expirations),
            repeat(atm_window_range)
        ):
            all_regs.extend(regs)
    return pd.DataFrame(all_regs)

# =========================
# ESTRATEGIAS (con POP %)
# =========================
def put_credit_spread(df, width_range, delta_range, credit_range):
    res = []
    puts = df[df["OptionType"]=="put"]
    for (sym, exp), grp in puts.groupby(["Ticker","Expiración"]):
        shorts = grp[grp["Delta"].between(delta_range[0], delta_range[1])]
        for _, s in shorts.iterrows():
            for _, l in grp.iterrows():
                w  = s.Strike - l.Strike
                mc = s.Mid - l.Mid
                if not (width_range[0] <= w <= width_range[1] and credit_range[0] <= mc <= credit_range[1]):
                    continue
                ret = (mc/w)/(s.Dias/30)*100
                res.append({
                    "Ticker": sym, "Expiración": exp, "Dias": s.Dias,
                    "Short Strike": s.Strike, "Long Strike": l.Strike,
                    "Delta": s.Delta, "POP (%)": _pop_from_delta(s.Delta),
                    "IV (%)": s["IV (%)"], "IV Rank": s["IV Rank"], "Cambio 1M (%)": s["Cambio 1M (%)"],
                    "Width": w, "Mid Credit": round(mc,2), "Return %": round(ret,2)
                })
    return pd.DataFrame(res)

def bear_call_spread(df, width_range, delta_range, credit_range):
    res = []
    calls = df[df["OptionType"]=="call"]
    for (sym, exp), grp in calls.groupby(["Ticker","Expiración"]):
        shorts = grp[grp["Delta"].between(delta_range[0], delta_range[1])]
        for _, s in shorts.iterrows():
            for _, l in grp.iterrows():
                w  = l.Strike - s.Strike
                mc = s.Mid - l.Mid
                if not (width_range[0] <= w <= width_range[1] and credit_range[0] <= mc <= credit_range[1]):
                    continue
                ret = (mc/w)/(s.Dias/30)*100
                res.append({
                    "Ticker": sym, "Expiración": exp, "Dias": s.Dias,
                    "Short Strike": s.Strike, "Long Strike": l.Strike,
                    "Delta": s.Delta, "POP (%)": _pop_from_delta(s.Delta),
                    "IV (%)": s["IV (%)"], "IV Rank": s["IV Rank"], "Cambio 1M (%)": s["Cambio 1M (%)"],
                    "Width": w, "Mid Credit": round(mc,2), "Return %": round(ret,2)
                })
    return pd.DataFrame(res)

def iron_condor(df, w_put_range, d_put_range, w_call_range, d_call_range, credit_range):
    pc = put_credit_spread(df, w_put_range, d_put_range, (credit_range[0], float('inf')))
    bc = bear_call_spread(df, w_call_range, d_call_range, (credit_range[0], float('inf')))
    res = []
    for _, p in pc.iterrows():
        matches = bc[(bc.Ticker==p.Ticker)&(bc.Expiración==p.Expiración)]
        for _, c in matches.iterrows():
            tot = p["Mid Credit"] + c["Mid Credit"]
            if not (credit_range[0] <= tot <= credit_range[1]):
                continue
            ret = (tot/(p.Width + c.Width))/(p.Dias/30)*100
            res.append({
                "Ticker": p.Ticker, "Expiración": p.Expiración, "Dias": p.Dias,
                "Put Short Strike": p["Short Strike"], "Put Long Strike": p["Long Strike"],
                "Call Short Strike": c["Short Strike"], "Call Long Strike": c["Long Strike"],
                "Delta Put": p.Delta, "POP Put (%)": _pop_from_delta(p.Delta),
                "Delta Call": c.Delta, "POP Call (%)": _pop_from_delta(c.Delta),
                "IV (%)": p["IV (%)"], "IV Rank": p["IV Rank"], "Cambio 1M (%)": p["Cambio 1M (%)"],
                "Width Put": p.Width, "Width Call": c.Width,
                "Mid Credit Total": round(tot,2), "Return %": round(ret,2)
            })
    return pd.DataFrame(res)

def iron_fly(df, width_range, delta_range, credit_range):
    res = []
    calls = df[df["OptionType"]=="call"]  # usamos call central para simetría
    for (sym, exp), grp in calls.groupby(["Ticker","Expiración"]):
        shorts = grp[grp["Delta"].between(delta_range[0], delta_range[1])]
        for _, s in shorts.iterrows():
            for w in range(width_range[0], width_range[1]+1):
                low  = grp[grp.Strike == s.Strike - w]
                high = grp[grp.Strike == s.Strike + w]
                if low.empty or high.empty:
                    continue
                mc = 2*s.Mid - low.Mid.iloc[0] - high.Mid.iloc[0]
                if not (credit_range[0] <= mc <= credit_range[1]):
                    continue
                ret = (mc/(2*w))/(s.Dias/30)*100
                res.append({
                    "Ticker": sym, "Expiración": exp, "Dias": s.Dias,
                    "Central Strike": s.Strike, "Width": w,
                    "Delta": s.Delta, "POP (%)": _pop_from_delta(s.Delta),
                    "IV (%)": s["IV (%)"], "IV Rank": s["IV Rank"], "Cambio 1M (%)": s["Cambio 1M (%)"],
                    "Mid Credit": round(mc,2), "Return %": round(ret,2)
                })
    return pd.DataFrame(res)

def jade_lizard(df, w_call_range, d_put_range, d_call_range, credit_range):
    res = []
    for (sym, exp), grp in df.groupby(["Ticker","Expiración"]):
        sp = grp[(grp.OptionType=="put")  & (grp.Delta.between(d_put_range[0], d_put_range[1]))]
        sc = grp[(grp.OptionType=="call") & (grp.Delta.between(d_call_range[0], d_call_range[1]))]
        for _, p in sp.iterrows():
            for _, c in sc.iterrows():
                for w in range(w_call_range[0], w_call_range[1]+1):
                    lc_strike = c.Strike + w
                    long_call = grp[(grp.OptionType=="call") & (grp.Strike==lc_strike)]
                    if long_call.empty:
                        continue
                    mc  = p.Mid + c.Mid - long_call.Mid.iloc[0]
                    if not (credit_range[0] <= mc <= credit_range[1]):
                        continue
                    ret = (mc/max(w,1))/(p.Dias/30)*100
                    res.append({
                        "Ticker": sym, "Expiración": exp, "Dias": p.Dias,
                        "Short Put": p.Strike, "Short Call": c.Strike, "Long Call": lc_strike,
                        "Delta Put": p.Delta, "POP Put (%)": _pop_from_delta(p.Delta),
                        "Delta Call": c.Delta, "POP Call (%)": _pop_from_delta(c.Delta),
                        "IV (%)": p["IV (%)"], "IV Rank": p["IV Rank"], "Cambio 1M (%)": p["Cambio 1M (%)"],
                        "Mid Credit": round(mc,2), "Return %": round(ret,2)
                    })
    return pd.DataFrame(res)

# =========================
# UI STREAMLIT
# =========================
st.set_page_config(page_title="Filtro Estrategias Opciones", layout="wide")
st.title("⚡ Filtro Base + Estrategias (S&P 500)")

# 0) Tickers con "Seleccionar todos"
all_tickers = obtener_tickers_sp500()
select_all = st.sidebar.checkbox("Seleccionar todos los tickers", value=True)
selected_tickers = all_tickers if select_all else st.sidebar.multiselect("Tickers", all_tickers)

# 0b) Tipos a incluir al cargar (por defecto SOLO puts para acelerar)
option_types_to_load = st.sidebar.multiselect(
    "Tipos de opción a incluir al cargar", ["put", "call"], default=["put"], key="k_types_load"
)
if not option_types_to_load:
    st.sidebar.warning("Selecciona al menos un tipo (put/call) para cargar.")

# --- PRESET ---
st.sidebar.header("Preset")
if st.sidebar.button("Aplicar preset (20–45 DTE, Δ −0.30 a +0.30, IV ≥ 25%, IVR ≥ 30%)"):
    st.session_state["k_dias"]   = (20, 45)
    st.session_state["k_delta"]  = (-0.30, 0.30)
    st.session_state["k_iv"]     = (25.0, 100.0)
    st.session_state["k_ivr"]    = (30.0, 100.0)
    st.rerun()

# 1) Configurar base
st.sidebar.header("1. Configurar Base")
r_dias  = st.sidebar.slider("Días hasta expiración", 1, 60, st.session_state.get("k_dias", (1, 60)), key="k_dias")
max_exp = st.sidebar.number_input("Máx. expiraciones por ticker (0 = sin límite)", min_value=0, max_value=30, value=0, step=1, key="k_maxexp")
# Ventana ATM como rango (min%, max%) – por defecto sin limitar (0,100)
atm_win = st.sidebar.slider("Ventana ATM (min%, max%)", 0, 100, (0, 100), help="Mantiene strikes con |K−S|/S*100 dentro de este rango.")

# Filtros visuales base (también aplicables a la carga por días)
r_ret   = st.sidebar.slider("Retorno mensual (%)", -100.0, 100.0, st.session_state.get("k_ret", (-100.0, 100.0)), key="k_ret")
r_dlt   = st.sidebar.slider("Delta", -1.0, 1.0, st.session_state.get("k_delta", (-1.0, 1.0)), key="k_delta")
r_iv    = st.sidebar.slider("IV (%)", 0.0, 100.0, st.session_state.get("k_iv", (0.0, 100.0)), key="k_iv")
r_ir    = st.sidebar.slider("IV Rank", 0.0, 100.0, st.session_state.get("k_ivr", (0.0, 100.0)), key="k_ivr")
r_ch    = st.sidebar.slider("Cambio 1M (%)", -100.0, 100.0, st.session_state.get("k_ch", (-100.0, 100.0)), key="k_ch")
workers = st.sidebar.number_input("Hilos (workers)", min_value=2, max_value=50, value=20, key="k_workers")

if st.sidebar.button("🔄 Cargar base") and option_types_to_load:
    with st.spinner(f"Cargando {len(selected_tickers)} tickers, tipos {option_types_to_load}, días {r_dias}, ATM {atm_win}…"):
        df_base = cargar_base(selected_tickers, option_types_to_load, r_dias, max_exp, atm_win, max_workers=workers)
    st.session_state["base_df"] = df_base
    st.success(f"Base cargada: {len(df_base)} contratos")

# 2) Mostrar/filtrar base
if "base_df" in st.session_state:
    base = st.session_state["base_df"]
    # Filtro de visualización por tipo (independiente de lo cargado)
    if set(base.OptionType.unique()) == set(["put","call"]):
        tipos_vista = st.sidebar.multiselect("Tipos a mostrar", ["put","call"], default=["put","call"], key="k_types_view")
    else:
        tipos_vista = list(base.OptionType.unique())

    df = base[
        base["OptionType"].isin(tipos_vista) &
        base["Dias"].between(r_dias[0], r_dias[1]) &
        base["Retorno %"].between(r_ret[0], r_ret[1]) &
        base["Delta"].between(r_dlt[0], r_dlt[1]) &
        base["IV (%)"].between(r_iv[0], r_iv[1]) &
        base["IV Rank"].between(r_ir[0], r_ir[1]) &
        base["Cambio 1M (%)"].between(r_ch[0], r_ch[1])
    ]
    st.subheader(f"🔖 Base filtrada: {len(df)} contratos")
    st.dataframe(df, use_container_width=True)

    # 3) Estrategias
    st.sidebar.header("2. Estrategias")
    strat = st.sidebar.selectbox("Selecciona estrategia", [
        "Put credit spread", "Bear call spread", "Iron Condor",
        "Iron Fly", "Jade Lizard", "Broken Wing Butterfly"
    ], key="k_strat")

    w_range    = st.sidebar.slider("Width range", 1, 20, (1, 5), key="k_width")
    cred_range = st.sidebar.slider("Mid Credit range", -10.0, 10.0, (-1.0, 1.0), key="k_credit")

    out = pd.DataFrame()
    if strat == "Put credit spread":
        d_range = st.sidebar.slider("Delta short put", -1.0, 0.0, st.session_state.get("k_dsp", (-0.30, -0.15)), key="k_dsp")
        if st.sidebar.button("Aplicar estrategia"): out = put_credit_spread(df, w_range, d_range, cred_range)
    elif strat == "Bear call spread":
        d_range = st.sidebar.slider("Delta short call", 0.0, 1.0, st.session_state.get("k_dsc", (0.15, 0.30)), key="k_dsc")
        if st.sidebar.button("Aplicar estrategia"): out = bear_call_spread(df, w_range, d_range, cred_range)
    elif strat == "Iron Condor":
        dp = st.sidebar.slider("Delta short put", -1.0, 0.0, st.session_state.get("k_dsp_ic", (-0.30, -0.15)), key="k_dsp_ic")
        dc = st.sidebar.slider("Delta short call", 0.0, 1.0, st.session_state.get("k_dsc_ic", (0.15, 0.30)), key="k_dsc_ic")
        if st.sidebar.button("Aplicar estrategia"): out = iron_condor(df, w_range, dp, w_range, dc, cred_range)
    elif strat == "Iron Fly":
        d_range = st.sidebar.slider("Delta central", -1.0, 1.0, st.session_state.get("k_dfly", (-0.15, 0.15)), key="k_dfly")
        if st.sidebar.button("Aplicar estrategia"): out = iron_fly(df, w_range, d_range, cred_range)
    elif strat == "Jade Lizard":
        dp = st.sidebar.slider("Delta short put", -1.0, 0.0, st.session_state.get("k_dsp_j", (-0.30, -0.15)), key="k_dsp_j")
        dc = st.sidebar.slider("Delta short call", 0.0, 1.0, st.session_state.get("k_dsc_j", (0.15, 0.30)), key="k_dsc_j")
        if st.sidebar.button("Aplicar estrategia"): out = jade_lizard(df, w_range, dp, dc, cred_range)
    else:  # Broken Wing Butterfly (MVP simple reutilizando iron_fly)
        d_range = st.sidebar.slider("Delta central", -1.0, 1.0, st.session_state.get("k_dbwb", (-0.15, 0.15)), key="k_dbwb")
        if st.sidebar.button("Aplicar estrategia"): out = iron_fly(df, (-w_range[1], w_range[1]), d_range, cred_range)

    if not out.empty:
        st.subheader(f"📊 {strat}: {len(out)} oportunidades")
        st.dataframe(out, use_container_width=True)
    else:
        st.caption("Ajusta parámetros y pulsa 'Aplicar estrategia' para ver resultados.")
