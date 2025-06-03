
# app_test.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
from datetime import datetime
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor

# Tradier API Config
TRADIER_TOKEN = "aLvdjAoMpkuiGLVzC1hgaAAGIv9I"
HEADERS = {"Authorization": f"Bearer {TRADIER_TOKEN}", "Accept": "application/json"}
BASE_URL = "https://api.tradier.com/v1"

# App Title
st.title("⚡ Filtro Rápido de Venta de Puts (S&P 500)")

# Función para calcular delta usando Black-Scholes
def calcular_delta(S, K, T, r, sigma):
    if sigma == 0 or T == 0:
        return -1.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1

# Obtener tickers del S&P 500 desde Wikipedia
@st.cache_data
def obtener_tickers_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)
    return sorted(table[0]["Symbol"].tolist())

tickers_sp500 = obtener_tickers_sp500()

# Función para obtener expiraciones
def get_expirations(symbol):
    try:
        url = f"{BASE_URL}/markets/options/expirations"
        r = requests.get(url, headers=HEADERS, params={"symbol": symbol})
        expirations = r.json().get("expirations", {}).get("date", [])
        return expirations
    except Exception:
        return []

# Función para obtener cadena de opciones
def get_option_chain(symbol, expiration):
    try:
        url = f"{BASE_URL}/markets/options/chains"
        r = requests.get(url, headers=HEADERS, params={"symbol": symbol, "expiration": expiration, "greeks": "true"})
        return r.json().get("options", {}).get("option", [])
    except Exception:
        return []

# Función para obtener precio actual
def get_quote(symbol):
    try:
        url = f"{BASE_URL}/markets/quotes"
        r = requests.get(url, headers=HEADERS, params={"symbols": symbol})
        data = r.json().get("quotes", {}).get("quote")
        return data[0] if isinstance(data, list) else data
    except Exception:
        return {}

# Función para calcular IV Rank aproximado
def calcular_iv_rank(ticker):
    try:
        data = yf.Ticker(ticker).history(period="1y")
        ivs = data["Close"].pct_change().rolling(window=21).std() * np.sqrt(252)
        if ivs.dropna().empty:
            return None
        iv_actual = ivs.iloc[-1]
        iv_rank = (iv_actual - ivs.min()) / (ivs.max() - ivs.min())
        return round(iv_rank * 100, 2)
    except:
        return None

# Procesar cada ticker individualmente
def procesar_ticker(ticker, dias_max):
    resultado = []
    quote = get_quote(ticker)
    if not quote or quote.get("last") is None:
        return resultado
    price = quote["last"]
    expirations = get_expirations(ticker)
    iv_rank = calcular_iv_rank(ticker)
    for exp_date in expirations:
        try:
            dias = (datetime.strptime(exp_date, "%Y-%m-%d") - datetime.now()).days
            if dias > dias_max:
                continue
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
                T = dias / 365
                if delta is None:
                    if iv is None:
                        continue
                    delta = calcular_delta(price, strike, T, 0.04, iv)
                retorno_mensual = (mid / strike) / (dias / 30)
                resultado.append({
                    "Ticker": ticker,
                    "Expiración": exp_date,
                    "Strike": strike,
                    "Bid": bid,
                    "Ask": ask,
                    "Mid Price": round(mid, 2),
                    "Delta": round(delta, 3),
                    "IV": round(iv, 3) if iv else "N/D",
                    "IV Rank": iv_rank if iv_rank is not None else "N/D",
                    "DTE": dias,
                    "Retorno mensual %": round(retorno_mensual * 100, 2)
                })
        except:
            continue
    return resultado

# Ejecutar procesamiento en paralelo
def procesar_tickers(tickers, dias_max):
    resultados = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(procesar_ticker, ticker, dias_max) for ticker in tickers]
        for future in futures:
            resultados.extend(future.result())
    return pd.DataFrame(resultados)

# App principal
st.sidebar.title("📊 Filtros para buscar puts")
dias_max = st.sidebar.slider("Días hasta expiración (máximo)", 5, 60, 30)
retorno = st.sidebar.slider("Retorno mensual %", 0.0, 10.0, (1.0, 5.0))
delta = st.sidebar.slider("Delta", -1.0, -0.01, (-0.30, -0.10))
iv = st.sidebar.slider("IV", 0.0, 2.0, (0.2, 1.5))
ivrank = st.sidebar.slider("IV Rank", 0.0, 100.0, (0.0, 100.0))
modo_debug = st.sidebar.checkbox("📋 Mostrar modo debug (sin filtros)")

# Botón para actualizar base de datos
if st.button("🔄 Actualizar base de datos"):
    df_global = procesar_tickers(tickers_sp500, dias_max)
    df_global.to_csv("cached_data.csv", index=False)
    st.success("✅ Base de datos actualizada.")

# Mostrar datos si ya hay archivo guardado
if os.path.exists("cached_data.csv"):
    df_global = pd.read_csv("cached_data.csv")
    df_global["IV"] = pd.to_numeric(df_global["IV"], errors="coerce")
    df_global["Delta"] = pd.to_numeric(df_global["Delta"], errors="coerce")
    df_global["Retorno mensual %"] = pd.to_numeric(df_global["Retorno mensual %"], errors="coerce")
    df_global["IV Rank"] = pd.to_numeric(df_global["IV Rank"], errors="coerce")
    df_global["DTE"] = pd.to_numeric(df_global["DTE"], errors="coerce")

    if not modo_debug:
        df_global = df_global[
            (df_global["DTE"] <= dias_max) &
            (df_global["Retorno mensual %"] >= retorno[0]) & (df_global["Retorno mensual %"] <= retorno[1]) &
            (df_global["Delta"] >= delta[0]) & (df_global["Delta"] <= delta[1]) &
            (df_global["IV"] >= iv[0]) & (df_global["IV"] <= iv[1]) &
            (df_global["IV Rank"] >= ivrank[0]) & (df_global["IV Rank"] <= ivrank[1])
        ]

    st.write(f"🔍 Se encontraron {len(df_global)} puts que cumplen los filtros:")
    st.dataframe(df_global.reset_index(drop=True), use_container_width=True)
else:
    st.warning("Primero debes actualizar la base de datos.")
