import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet

# Konfigurasi Halaman Web
st.set_page_config(page_title="IHSG Quant & AI Screener", layout="wide")

# --- JUDUL & DISCLAIMER ---
st.title("📈 IHSG Quant & AI Screener")
st.markdown("Cari saham *undervalue*, analisis grafiknya, dan prediksi harga 30 hari ke depan menggunakan Machine Learning.")

with st.expander("ℹ️ Tentang Sistem & Disclaimer Data", expanded=False):
    st.write("""
    * **Sumber Data:** Data historis dan fundamental pasar ditarik secara *real-time* dari **Yahoo Finance (`yfinance`)**.
    * **Analisis & AI:** Sistem memproses indikator teknikal (RSI, MACD, MA200, Bollinger Bands) serta peramalan tren menggunakan algoritma Machine Learning (**Prophet**).
    * **Disclaimer:** Aplikasi ini dirancang sebagai alat bantu riset mandiri (*value investing*). Seluruh keputusan transaksi jual/beli saham berada sepenuhnya di tangan pengguna.
    """)

st.markdown("---")

# --- DAFTAR SAHAM UNGGULAN IHSG (LQ45 / PILIHAN UTAMA) ---
DEFAULT_IHSG_TICKERS = [
    "ADRO.JK", "AKRA.JK", "AMMN.JK", "AMRT.JK", "ANTM.JK", "ASII.JK", 
    "BBCA.JK", "BBNI.JK", "BBRI.JK", "BBTN.JK", "BMRI.JK", "BRPT.JK", 
    "CPIN.JK", "EXCL.JK", "GOTO.JK", "ICBP.JK", "INCO.JK", "INDF.JK", 
    "INKP.JK", "ISAT.JK", "ITMG.JK", "JPFA.JK", "KLBF.JK", "MAPI.JK", 
    "MBMA.JK", "MDKA.JK", "MEDC.JK", "PGAS.JK", "PGEO.JK", "PTBA.JK", 
    "SMGR.JK", "TLKM.JK", "TOWR.JK", "UNTR.JK", "UNVR.JK", "ARTO.JK"
]

# --- FUNGSI INDIKATOR TEKNIKAL ---
def calculate_indicators(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    return df

# --- SIDEBAR PENGATURAN ---
st.sidebar.header("⚙️ Pengaturan Screener")

use_all_market = st.sidebar.checkbox("🌐 Gunakan Semua Saham Unggulan IHSG", value=False)

if use_all_market:
    ticker_input = ", ".join(DEFAULT_IHSG_TICKERS)
    st.sidebar.info("Mode aktif: Memindai daftar saham unggulan bursa Indonesia.")
else:
    ticker_input = st.sidebar.text_input(
        "Daftar Saham (Pisahkan koma)", 
        "BBRI.JK, BMRI.JK, ASII.JK, TLKM.JK, BBCA.JK",
        help="Contoh: BBRI.JK, BMRI.JK (Gunakan akhiran .JK untuk emiten bursa Indonesia)."
    )

# --- LOGIKA UTAMA SCREENER ---
if st.sidebar.button("🔍 Mulai Screening", type="primary"):
    tickers = [t.strip().upper() for t in ticker_input.split(',')]
    progress_bar = st.progress(0)
    results = []
    
    st.session_state['stock_data'] = {} 
    
    for i, ticker in enumerate(tickers):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            per = info.get('trailingPE', 0)
            pbv = info.get('priceToBook', 0)
            
            # Ambil Dividend Yield & ROE (normalisasi ke %)
            div_yield = info.get('dividendYield', 0)
            if div_yield and div_yield < 1:
                div_yield = div_yield * 100
            div_yield = div_yield if div_yield else 0
            
            roe = info.get('returnOnEquity', 0)
            if roe and roe < 1:
                roe = roe * 100
            roe = roe if roe else 0
            
            df = yf.download(ticker, period="2y", interval="1d", progress=False)
            
            if not df.empty and len(df) > 50:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                    
                df = calculate_indicators(df)
                df.dropna(inplace=True)
                
                if not df.empty:
                    st.session_state['stock_data'][ticker] = df
                    
                    latest = df.iloc[-1]
                    close_price = float(latest['Close'])
                    rsi = float(latest['RSI_14']) if 'RSI_14' in latest else 50
                    macd_line = float(latest['MACD']) if 'MACD' in latest else 0
                    signal_line = float(latest['MACD_Signal']) if 'MACD_Signal' in latest else 0
                    ma200 = float(latest['SMA_200']) if 'SMA_200' in latest else close_price
                    
                    trend = "UPTREND" if close_price > ma200 else "DOWNTREND"
                    
                    if trend == "UPTREND" and macd_line > signal_line:
                        signal = "🟢 AKUMULASI (BUY)"
                    else:
                        signal = "⏳ WAIT & SEE"
                        
                    results.append({
                        "Saham / Emiten": ticker.replace(".JK", ""),
                        "Harga (Rp)": round(close_price, 2),
                        "Deviden Yield (%)": round(div_yield, 2),
                        "ROE (%)": round(roe, 2),
                        "PER (x)": round(per, 2) if per else 0,
                        "PBV (x)": round(pbv, 2) if pbv else 0,
                        "RSI": round(rsi, 2),
                        "Tren Mayor": trend,
                        "Sinyal": signal
                    })
        except Exception as e:
            pass
            
        progress_bar.progress((i + 1) / len(tickers))
    
    st.session_state['results'] = results

# --- TAMPILAN UTAMA (3 TAB) ---
if 'results' in st.session_state and st.session_state['results']:
    tab1, tab2, tab3 = st.tabs(["📋 Tabel Screener", "📊 Grafik Analisis", "🤖 Prediksi AI (30 Hari)"])
    
    with tab1:
        st.subheader("Hasil Screening & Valuasi Saham")
        st.caption("💡 Klik pada judul kolom tabel (misal: PER, PBV, ROE, dll) untuk mengurutkan data secara interaktif.")
        
        df_result = pd.DataFrame(st.session_state['results'])
        # Tambahkan kolom Nomor urut otomatis
        df_result.insert(0, 'Nomor', range(1, len(df_result) + 1))
        
        st.dataframe(df_result, use_container_width=True, hide_index=True)
        
        # --- KOTAK PANDUAN EDUKATIF DI BAWAH TABEL ---
        with st.expander("📖 Panduan & Standar Penilaian Indikator untuk Value Investing", expanded=True):
            st.markdown("""
            * **Deviden Yield (%):** Persentase pembagian dividen tunai terhadap harga saham. **Semakin tinggi semakin baik**, menunjukkan perusahaan royal kepada pemegang saham.
            * **ROE (%):** Tingkat pengembalian ekuitas (*Return on Equity*). **Semakin tinggi semakin baik** (ideal >10%-15%), mencerminkan efisiensi manajemen mencetak laba.
            * **PER (x):** Perbandingan harga terhadap laba bersih. **Semakin kecil semakin baik** (standar murah di bawah 15x atau 10x).
            * **PBV (x):** Perbandingan harga terhadap nilai buku aset bersih. **Semakin kecil semakin baik** (di bawah 1x berarti harga diskon di bawah nilai bukunya).
            * **RSI:** Indikator momentum (0–100). Di bawah 30 menandakan *Oversold* (area menarik untuk akumulasi), di atas 70 *Overbought* (jenuh beli).
            * **Tren Mayor:** Arah tren jangka panjang berdasarkan MA200 (*UPTREND* lebih aman untuk investasi menengah).
            """)
        
    with tab2:
        st.subheader("Grafik Analisis Teknikal Interaktif")
        available_tickers = list(st.session_state['stock_data'].keys())
        selected_ticker_chart = st.selectbox("Pilih Saham:", available_tickers, key="chart_select")
        
        if selected_ticker_chart:
            df_plot = st.session_state['stock_data'][selected_ticker_chart]
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])

            fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name='Harga'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA_200'], line=dict(color='orange', width=2), name='MA 200'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI_14'], line=dict(color='purple', width=2), name='RSI'), row=2, col=1)
            
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

            fig.update_layout(title=f"Pergerakan Harga & RSI - {selected_ticker_chart}", height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
    with tab3:
        st.subheader("Prediksi Pergerakan Harga dengan AI (Prophet)")
        st.warning("⚠️ Disclaimer: Prediksi Machine Learning didasarkan pada probabilitas tren historis dan BUKAN jaminan harga masa depan.")
        
        available_tickers_ml = list(st.session_state['stock_data'].keys())
        selected_ticker_ml = st.selectbox("Pilih Saham untuk Prediksi AI:", available_tickers_ml, key="ml_select")
        
        if selected_ticker_ml:
            with st.spinner("Model AI sedang mempelajari pola historis dan menghitung proyeksi 30 hari..."):
                df_ml = st.session_state['stock_data'][selected_ticker_ml].copy()
                df_prophet = df_ml.reset_index()[['Date', 'Close']]
                df_prophet.columns = ['ds', 'y']
                df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
                
                model = Prophet(daily_seasonality=False, yearly_seasonality=True)
                model.fit(df_prophet)
                
                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)
                
                last_prediction = forecast.iloc[-1]
                current_price = df_prophet['y'].iloc[-1]
                predicted_price = last_prediction['yhat']
                trend_direction = "NAIK 🟢" if predicted_price > current_price else "TURUN 🔴"
                
                st.success(f"Prediksi Selesai! Estimasi tren 30 hari ke depan untuk **{selected_ticker_ml}**: **{trend_direction}** (Target AI: Rp{predicted_price:,.0f})")
                
                # Grafik Peramalan AI (Langsung tampil otomatis di bawah kotak hasil)
                fig_prophet = go.Figure()
                fig_prophet.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Data Historis', line=dict(color='cyan')))
                fig_prophet.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Prediksi AI (Trend 30 Hari)', line=dict(color='orange', dash='dash')))
                
                fig_prophet.update_layout(
                    title=f"Forecast Harga 30 Hari - {selected_ticker_ml}", 
                    yaxis_title="Harga (Rp)", 
                    xaxis_title="Tanggal", 
                    template="plotly_dark", 
                    height=500
                )
                st.plotly_chart(fig_prophet, use_container_width=True)

elif 'results' in st.session_state and not st.session_state['results']:
    st.info("Tidak ada data emiten yang ditemukan. Silakan klik 'Mulai Screening' di sidebar.")
else:
    st.info("👈 Silakan masukkan daftar saham atau centang pilihan pasar di *sidebar* kiri, lalu klik **Mulai Screening** untuk memulai analisis.")