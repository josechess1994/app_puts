import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import concurrent.futures
from datetime import datetime
from scipy.stats import norm
import yfinance as yf

# ========= Token seguro (st.secrets o variable de entorno) =========
# Opción A: Streamlit Cloud -> añade TRADIER_TOKEN en "Settings > Secrets"
# Opción B: Local -> crea un .env con TRADIER_TOKEN=... y exporta la variable
try:
    from dotenv import load_dotenv  # opcional si usas .env en local
    load_dotenv()
except Exception:
    pass

TOKEN = (getattr(st, "secrets", {}).get("TRADIER_TOKEN")
         if hasattr(st, "secrets") else None) or os.getenv("TRADIER_TOKEN")

if not TOKEN:
    st.set_page_config(page_title="Filtro Estrategias Opciones", layout="wide")
    st.error("❌ Falta TRADIER_TOKEN. Ponlo en Streamlit Secrets o como variable de entorno.")
    st.stop()

# --- CONFIGURACIÓN ---
BASE_URL = "https://api.tradier.com/v1"
HEADERS  = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/json"}

# --- AUXILIARES ---
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

# --- CÁLCULOS GRIEGOS ---
def calcular_delta(S, K, T, r, sigma):
    if sigma is None or sigma == 0 or T == 0:
        return None
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1) - 1

def black_scholes_put_iv(S, K, T, price=0.01, r=0.04):
    eps, sigma = 1e-5, 0.3
    for _ in range(100):
        d1 = (np.log(S/K) + (r+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        model = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        vega  = max(S*np.sqrt(T)*norm.pdf(d1), 1e-8)
        diff  = model - price
        if abs(diff) < eps:
            return sigma
        sigma = max(sigma - diff/vega, 1e-3)
    return sigma

def calcular_iv_rank(ticker):
    try:
        closes = yf.Ticker(ticker).history(period="1y")["Close"]
        ivs    = closes.pct_change().rolling(21).std() * np.sqrt(252)
        act    = ivs.iloc[-1]
        return round((act - ivs.min())/(ivs.max() - ivs.min())*100, 2)
    except:
        return None

# --- DATOS OPCIONES ---
def get_expirations(symbol):
    try:
        r = requests.get(f"{BASE_URL}/markets/options/expirations", headers=HEADERS, params={"symbol": symbol}, timeout=10)
        return r.json().get("expirations", {}).get("date", [])
    except:
        return []

def get_option_chain(symbol, expiration):
    try:
        r = requests.get(f"{BASE_URL}/markets/options/chains", headers=HEADERS,
                         params={"symbol": symbol, "expiration": expiration, "greeks": "true"}, timeout=10)
        return r.json().get("options", {}).get("option", [])
    except:
        return []

def get_quote(symbol):
    try:
        r = requests.get(f"{BASE_URL}/markets/quotes", headers=HEADERS, params={"symbols": symbol}, timeout=10)
        data = r.json().get("quotes", {}).get("quote")
        return data[0] if isinstance(data, list) else data or {}
    except:
        return {}

# --- CONSTRUCCIÓN BASE ---
def procesar_ticker(ticker):
    registros = []
    q = get_quote(ticker)
    last = q.get("last")
    if last is None:
        return registros
    ivr = calcular_iv_rank(ticker)
    cambio = obtener_cambio_mensual(ticker)
    for exp in get_expirations(ticker):
        dias = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days
        if dias <= 0:
            continue
        T = dias/365
        for o in get_option_chain(ticker, exp):
            bid = o.get("bid") or 0
            ask = o.get("ask") or 0
            mid = (bid+ask)/2
            if mid <= 0:
                continue
            greeks = o.get("greeks") or {}
            iv_dec = greeks.get("mid_iv") or black_scholes_put_iv(last, o["strike"], T, mid)
            iv_pct = iv_dec*100 if iv_dec is not None else None
            delta = greeks.get("delta") or calcular_delta(last, o["strike"], T, 0.04, iv_dec)
            ret = (mid/o["strike"])/(dias/30)*100
            registros.append({
                "Ticker": ticker,
                "Expiración": exp,
                "OptionType": o.get("option_type"),
                "Dias": dias,
                "Strike": o.get("strike"),
                "Mid": round(mid,4),
                "Retorno %": round(ret,2),
                "Delta": round(delta or 0,3),
                "IV (%)": round(iv_pct,2) if iv_pct is not None else None,
                "IV Rank": ivr,
                "Cambio 1M (%)": cambio
            })
    return registros

def cargar_base(tickers, workers):
    all_recs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for recs in executor.map(procesar_ticker, tickers):
            all_recs.extend(recs)
    return pd.DataFrame(all_recs)

# --- ESTRATEGIAS ---
def put_credit_spread(df, width_range, delta_range, credit_range):
    rows=[]
    puts=df[df.OptionType=="put"]
    for (sym,exp),grp in puts.groupby(["Ticker","Expiración"]):
        shorts=grp[grp.Delta.between(delta_range[0],delta_range[1])]
        for _,s in shorts.iterrows():
            for _,l in grp.iterrows():
                w=s.Strike-l.Strike
                mc=s.Mid-l.Mid
                if not(width_range[0]<=w<=width_range[1] and credit_range[0]<=mc<=credit_range[1]):
                    continue
                ret=(mc/w)/(s.Dias/30)*100
                rows.append({
                    "Ticker":sym,"Expiración":exp,"Dias":s.Dias,
                    "Short Strike":s.Strike,"Long Strike":l.Strike,
                    "Delta":s.Delta,"IV (%)":s["IV (%)"],"IV Rank":s["IV Rank"],
                    "Cambio 1M (%)":s["Cambio 1M (%)"],"Width":w,
                    "Mid Credit":round(mc,2),"Return %":round(ret,2)
                })
    return pd.DataFrame(rows)

def bear_call_spread(df, width_range, delta_range, credit_range):
    rows=[]
    calls=df[df.OptionType=="call"]
    for (sym,exp),grp in calls.groupby(["Ticker","Expiración"]):
        shorts=grp[grp.Delta.between(delta_range[0],delta_range[1])]
        for _,s in shorts.iterrows():
            for _,l in grp.iterrows():
                w=l.Strike-s.Strike
                mc=s.Mid-l.Mid
                if not(width_range[0]<=w<=width_range[1] and credit_range[0]<=mc<=credit_range[1]):
                    continue
                ret=(mc/w)/(s.Dias/30)*100
                rows.append({
                    "Ticker":sym,"Expiración":exp,"Dias":s.Dias,
                    "Short Strike":s.Strike,"Long Strike":l.Strike,
                    "Delta":s.Delta,"IV (%)":s["IV (%)"],"IV Rank":s["IV Rank"],
                    "Cambio 1M (%)":s["Cambio 1M (%)"],"Width":w,
                    "Mid Credit":round(mc,2),"Return %":round(ret,2)
                })
    return pd.DataFrame(rows)

def iron_condor(df, w_put_range, d_put_range, w_call_range, d_call_range, credit_range):
    pc=put_credit_spread(df,w_put_range,d_put_range,credit_range)
    bc=bear_call_spread(df,w_call_range,d_call_range,credit_range)
    rows=[]
    for _,p in pc.iterrows():
        match=bc[(bc.Ticker==p.Ticker)&(bc.Expiración==p.Expiración)]
        for _,c in match.iterrows():
            tot=p["Mid Credit"]+c["Mid Credit"]
            if not(credit_range[0]<=tot<=credit_range[1]): continue
            ret=(tot/(p.Width+c.Width))/(p.Dias/30)*100
            rows.append({
                "Ticker":p.Ticker,"Expiración":p.Expiración,"Dias":p.Dias,
                "Put Short Strike":p["Short Strike"],"Put Long Strike":p["Long Strike"],
                "Call Short Strike":c["Short Strike"],"Call Long Strike":c["Long Strike"],
                "Delta Put":p.Delta,"Delta Call":c.Delta,"IV (%)":p["IV (%)"],
                "IV Rank":p["IV Rank"],"Cambio 1M (%)":p["Cambio 1M (%)"],
                "Width Put":p.Width,"Width Call":c.Width,
                "Mid Credit Total":round(tot,2),"Return %":round(ret,2)
            })
    return pd.DataFrame(rows)

def iron_fly(df, width_range, delta_range, credit_range):
    rows=[]
    calls=df[df.OptionType=="call"]
    for (sym,exp),grp in calls.groupby(["Ticker","Expiración"]):
        shorts=grp[grp.Delta.between(delta_range[0],delta_range[1])]
        for _,s in shorts.iterrows():
            for w in range(width_range[0],width_range[1]+1):
                lp=s.Strike-w
                hp=s.Strike+w
                low=grp[grp.Strike==lp]
                high=grp[grp.Strike==hp]
                if low.empty or high.empty: continue
                mc= s.Mid*2 - low.Mid.iloc[0] - high.Mid.iloc[0]
                if not(credit_range[0]<=mc<=credit_range[1]): continue
                ret=(mc/(2*w))/(s.Dias/30)*100
                rows.append({
                    "Ticker":sym,"Expiración":exp,"Dias":s.Dias,
                    "Central Strike":s.Strike,"Width":w,
                    "Delta":s.Delta,"IV (%)":s["IV (%)"],"IV Rank":s["IV Rank"],
                    "Cambio 1M (%)":s["Cambio 1M (%)"],
                    "Mid Credit":round(mc,2),"Return %":round(ret,2)
                })
    return pd.DataFrame(rows)

def jade_lizard(df, w_call_range, d_put_range, d_call_range, credit_range):
    rows=[]
    for (sym,exp),grp in df.groupby(["Ticker","Expiración"]):
        sp=grp[(grp.OptionType=="put")&(grp.Delta.between(d_put_range[0],d_put_range[1]))]
        sc=grp[(grp.OptionType=="call")&(grp.Delta.between(d_call_range[0],d_call_range[1]))]
        for _,p in sp.iterrows():
            for _,c in sc.iterrows():
                for w in range(w_call_range[0],w_call_range[1]+1):
                    lc=c.Strike+w
                    fc=grp[(grp.OptionType=="call")&(grp.Strike==lc)]
                    if fc.empty: continue
                    mc=p.Mid + c.Mid - fc.Mid.iloc[0]
                    if not(credit_range[0]<=mc<=credit_range[1]): continue
                    ret=(mc/w)/(p.Dias/30)*100
                    rows.append({
                        "Ticker":sym,"Expiración":exp,"Dias":p.Dias,
                        "Short Put":p.Strike,"Short Call":c.Strike,"Long Call":lc,
                        "Delta Put":p.Delta,"Delta Call":c.Delta,
                        "IV (%)":p["IV (%)"],"IV Rank":p["IV Rank"],
                        "Cambio 1M (%)":p["Cambio 1M (%)"],
                        "Mid Credit":round(mc,2),"Return %":round(ret,2)
                    })
    return pd.DataFrame(rows)

# --- INTERFAZ ---
st.set_page_config(page_title="Filtro Estrategias Opciones", layout="wide")
st.title("⚡ Filtro Base + Estrategias (S&P 500)")

# Selector opcional de tickers (agrega un checkbox para seleccionar todos)
all_tickers = obtener_tickers_sp500()
select_all = st.sidebar.checkbox("Seleccionar todos los tickers", value=True)
selected_tickers = all_tickers if select_all else st.sidebar.multiselect("Tickers", all_tickers, default=[])

# Parámetros base
st.sidebar.header("1. Configurar Base")
r_dias = st.sidebar.slider("Días exp",1,60,(1,60))
r_ret  = st.sidebar.slider("Retorno %",-100.0,100.0,(-100.0,100.0))
r_dlt  = st.sidebar.slider("Delta",-1.0,1.0,(-1.0,1.0))
r_iv   = st.sidebar.slider("IV (%)",0.0,100.0,(0.0,100.0))
r_ir   = st.sidebar.slider("IV Rank",0.0,100.0,(0.0,100.0))
r_ch   = st.sidebar.slider("Cambio 1M (%)",-100.0,100.0,(-100.0,100.0))
workers= st.sidebar.number_input("Hilos",2,50,20)
if st.sidebar.button("🔄 Cargar base"): 
    tickers = selected_tickers if (not select_all) else all_tickers
    df_base = cargar_base(tickers, workers)
    st.session_state["base_df"] = df_base
    st.success(f"Base cargada: {len(df_base)} contratos")

# Filtrar base
if "base_df" in st.session_state:
    base = st.session_state["base_df"]
    df = base[
        base.Dias.between(r_dias[0],r_dias[1]) &
        base["Retorno %"].between(r_ret[0],r_ret[1]) &
        base.Delta.between(r_dlt[0],r_dlt[1]) &
        base["IV (%)"].between(r_iv[0],r_iv[1]) &
        base["IV Rank"].between(r_ir[0],r_ir[1]) &
        base["Cambio 1M (%)"].between(r_ch[0],r_ch[1])
    ]
    st.subheader(f"🔖 Base filtrada: {len(df)} contratos")
    st.dataframe(df, use_container_width=True)

    # Estrategias
    st.sidebar.header("2. Estrategia")
    strat = st.sidebar.selectbox("Selecciona estrategia",[
        "Put credit spread","Bear call spread","Iron Condor",
        "Iron Fly","Jade Lizard","Broken Wing Butterfly"
    ])
    w_range = st.sidebar.slider("Width range",1,20,(1,5))
    cred_range = st.sidebar.slider("Mid Credit range",-10.0,10.0,(-1.0,1.0))

    if strat == "Put credit spread":
        d_range = st.sidebar.slider("Delta short put",-1.0,0.0,(-0.3,-0.15))
        if st.sidebar.button("Aplicar estrategia"): out = put_credit_spread(df,w_range,d_range,cred_range)
    elif strat == "Bear call spread":
        d_range = st.sidebar.slider("Delta short call",0.0,1.0,(0.15,0.3))
        if st.sidebar.button("Aplicar estrategia"): out = bear_call_spread(df,w_range,d_range,cred_range)
    elif strat == "Iron Condor":
        dp = st.sidebar.slider("Delta short put",-1.0,0.0,(-0.3,-0.15))
        dc = st.sidebar.slider("Delta short call",0.0,1.0,(0.15,0.3))
        if st.sidebar.button("Aplicar estrategia"): out = iron_condor(df,w_range,dp,w_range,dc,cred_range)
    elif strat == "Iron Fly":
        d_range = st.sidebar.slider("Delta short central",-1.0,1.0,(-0.15,0.15))
        if st.sidebar.button("Aplicar estrategia"): out = iron_fly(df,w_range,d_range,cred_range)
    elif strat == "Jade Lizard":
        dp = st.sidebar.slider("Delta short put",-1.0,0.0,(-0.3,-0.15))
        dc = st.sidebar.slider("Delta short call",0.0,1.0,(0.15,0.3))
        if st.sidebar.button("Aplicar estrategia"): out = jade_lizard(df,w_range,dp,dc,cred_range)
    else:  # Broken Wing Butterfly (usa iron_fly como base)
        d_range = st.sidebar.slider("Delta short central",-1.0,1.0,(-0.15,0.15))
        if st.sidebar.button("Aplicar estrategia"): out = iron_fly(df,(-w_range[1],w_range[1]),d_range,cred_range)

    if 'out' in locals():
        st.subheader(f"📊 {strat}: {len(out)} oportunidades")
        st.dataframe(out, use_container_width=True)
    else:
        st.info("Selecciona parámetros y haz clic en 'Aplicar estrategia'.")