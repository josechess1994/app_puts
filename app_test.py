# app_test_mejorado.py

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm
import yfinance as yf
import concurrent.futures
import os

BASE_URL = "https://api.tradier.com/v1"
TOKEN = "aLvdjAoMpkuiGLVzC1hgaAAGIv9I"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/json"}

@st.cache_data
def obtener_tickers_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tablas = pd.read_html(url)
    return tablas[0]["Symbol"].tolist()

@st.cache_data
def obtener_cambio_mensual(ticker):
    try:
        data = yf.Ticker(ticker).history(period="1mo")
        if len(data) >= 2:
            inicio = data["Close"].iloc[0]
            fin = data["Close"].iloc[-1]
            return round(((fin - inicio) / inicio) * 100, 2)
    except:
        pass
    return None

@st.cache_data
def obtener_fecha_earnings(ticker):
    try:
        cal = yf.Ticker(ticker).calendar
        if "Earnings Date" in cal.index:
            return pd.to_datetime(cal.loc["Earnings Date"][0]).date()
    except:
        return None

def calcular_delta(S, K, T, r, sigma):
    try:
        if sigma == 0 or T == 0:
            return -1.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1) - 1
    except:
        return None

def black_scholes_put_iv(S, K, T, price=0.01, r=0.04):
    epsilon = 1e-5
    max_iter = 100
    sigma = 0.3
    try:
        for _ in range(max_iter):
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            model_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            vega = S * np.sqrt(T) * norm.pdf(d1)
            diff = model_price - price
            if abs(diff) < epsilon:
                return sigma
            sigma -= diff / vega
            if sigma <= 0:
                sigma = 1e-3
    except:
        pass
    return None

def calcular_iv_rank(ticker):
    try:
        data = yf.Ticker(ticker).history(period="1y")
        ivs = data["Close"].pct_change().rolling(window=21).std() * np.sqrt(252)
        iv_actual = ivs.iloc[-1]
        iv_rank = (iv_actual - ivs.min()) / (ivs.max() - ivs.min())
        return round(iv_rank * 100, 2)
    except:
        return None

def get_expirations(symbol):
    url = f"{BASE_URL}/markets/options/expirations"
    params = {"symbol": symbol}
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data and "expirations" in data and "date" in data["expirations"]:
                return data["expirations"]["date"]
    except:
        pass
    return []

def get_option_chain(symbol, expiration):
    url = f"{BASE_URL}/markets/options/chains"
    params = {"symbol": symbol, "expiration": expiration, "greeks": "true"}
    try:
        r = requests.get(url, headers=HEADERS, params=params)
        if r.status_code == 200:
            return r.json().get("options", {}).get("option", [])
    except:
        pass
    return []

def get_quote(symbol):
    url = f"{BASE_URL}/markets/quotes"
    params = {"symbols": symbol}
    try:
        r = requests.get(url, headers=HEADERS, params=params)
        if r.status_code == 200:
            data = r.json().get("quotes", {}).get("quote")
            return data[0] if isinstance(data, list) else data
    except:
        pass
    return {}

def procesar_ticker(ticker, dias_maximos=120):
    resultados = []
    quote = get_quote(ticker)
    if not quote or quote.get("last") is None:
        return []
    price = quote["last"]
    expirations = get_expirations(ticker)
    iv_rank = calcular_iv_rank(ticker)
    cambio = obtener_cambio_mensual(ticker)
    earnings = obtener_fecha_earnings(ticker)
    for exp_date in expirations:
        try:
            dias = (datetime.strptime(exp_date, "%Y-%m-%d") - datetime.now()).days
            if dias > dias_maximos or dias <= 0:
                continue
            T = dias / 365
            opciones = get_option_chain(ticker, exp_date)
            for opt in opciones:
                if opt.get("option_type") != "put":
                    continue
                bid = opt.get("bid")
                ask = opt.get("ask")
                if not bid or not ask:
                    continue
                mid = (bid + ask) / 2
                strike = opt.get("strike")
                greeks = opt.get("greeks") or {}
                delta = greeks.get("delta")
                iv = greeks.get("mid_iv")
                if delta is None or iv is None:
                    iv = iv or black_scholes_put_iv(price, strike, T, mid)
                    delta = calcular_delta(price, strike, T, 0.04, iv)
                if delta is None or iv is None:
                    continue
                retorno_mensual = (mid / strike) / (dias / 30)
                resultados.append({
                    "Ticker": ticker,
                    "Expiración": exp_date,
                    "Strike": strike,
                    "Bid": bid,
                    "Ask": ask,
                    "Mid Price": round(mid, 2),
                    "Delta": round(delta, 3),
                    "IV": round(iv, 3),
                    "IV Rank": iv_rank if iv_rank is not None else "N/D",
                    "Retorno %": round(retorno_mensual * 100, 2),
                    "Dias hasta exp": dias,
                    "Cambio 1 mes %": cambio,
                    "Earnings": earnings
                })
        except:
            continue
    return resultados

# --- STREAMLIT ---
st.set_page_config(page_title="Filtro de Puts", layout="wide")
st.title("⚡ Filtro Rápido de Venta de Puts (S&P 500)")

st.sidebar.header("📡 Datos")
dias_maximos = st.sidebar.slider("Máx días hasta expiración", 30, 120, 60)

if st.sidebar.button("🔄 Actualizar base de datos"):
    tickers = obtener_tickers_sp500()
    st.info("Procesando 500 compañías... Esto puede tomar varios minutos.")
    resultados = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(procesar_ticker, t, dias_maximos) for t in tickers]
        for f in concurrent.futures.as_completed(futures):
            resultados.extend(f.result())
    df = pd.DataFrame(resultados)
    df.to_csv("puts_filtrados.csv", index=False)
    st.success(f"✅ Base de datos actualizada. Total puts: {len(df)}")

if os.path.exists("puts_filtrados.csv"):
    df = pd.read_csv("puts_filtrados.csv")
    df["IV"] = pd.to_numeric(df["IV"], errors="coerce")
    df["IV Rank"] = pd.to_numeric(df["IV Rank"], errors="coerce")
    df["Cambio 1 mes %"] = pd.to_numeric(df["Cambio 1 mes %"], errors="coerce")

    st.sidebar.header("🎛️ Filtros")
    dias = st.sidebar.slider("Días hasta expiración", 0, dias_maximos, (5, 35))
    retorno = st.sidebar.slider("Retorno mensual (%)", 0.0, 10.0, (1.0, 5.0))
    delta = st.sidebar.slider("Delta", -1.0, -0.01, (-0.4, -0.15))
    iv = st.sidebar.slider("IV", 0.0, 2.0, (0.2, 1.5))
    iv_rank = st.sidebar.slider("IV Rank", 0.0, 100.0, (0.0, 100.0))
    cambio = st.sidebar.slider("Cambio en 1 mes (%)", -100.0, 100.0, (-50.0, 50.0))
    modo_debug = st.sidebar.checkbox("🔧 Ver todo (sin filtros)")

    total_original = len(df)
    if not modo_debug:
        df = df[
            (df["Dias hasta exp"] >= dias[0]) & (df["Dias hasta exp"] <= dias[1]) &
            (df["Retorno %"] >= retorno[0]) & (df["Retorno %"] <= retorno[1]) &
            (df["Delta"] >= delta[0]) & (df["Delta"] <= delta[1]) &
            (df["IV"] >= iv[0]) & (df["IV"] <= iv[1]) &
            (df["IV Rank"] >= iv_rank[0]) & (df["IV Rank"] <= iv_rank[1]) &
            (df["Cambio 1 mes %"] >= cambio[0]) & (df["Cambio 1 mes %"] <= cambio[1])
        ]

    st.subheader(f"📊 {len(df)} puts filtrados (de {total_original} totales)")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar CSV", data=csv, file_name="puts_filtrados.csv")
else:
    st.warning("Primero debes actualizar la base de datos.")