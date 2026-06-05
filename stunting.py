import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

filename = 'model_stunting.pkl'
with open(filename, 'rb') as file:
        model = pickle.load(file)
df = pd.read_csv('stunting.csv')
list_kota = df['nama_kabupaten_kota'].unique()
list_kota = sorted(list_kota)
st.set_page_config(page_title="Prediksi Stunting", page_icon="👶", layout="wide")
st.sidebar.header("🎛️ Panel Input Data")
pilih_kota = st.sidebar.selectbox("Pilih Kabupaten/Kota:", list_kota)
st.sidebar.markdown("---")
st.sidebar.write("Masukkan Data Skenario Baru:")
avg_imunisasi = int(df['jumlah_imunisasi'].mean())
avg_sanitasi = int(df['jumlah_sanitasi'].mean())
avg_bblr = int(df['jumlah_bblr'].mean())
input_imunisasi = st.sidebar.number_input("💉 Balita Diimunisasi", min_value=0, value=avg_imunisasi)
input_sanitasi = st.sidebar.number_input("🚽 Sanitasi Layak", min_value=0, value=avg_sanitasi)
input_bblr = st.sidebar.number_input("⚖ Kasus BBLR", min_value=0, value=avg_bblr)
st.title(f"👶 Analisis Stunting: {pilih_kota}")
st.markdown(f"Tren historis stunting di **{pilih_kota}** (2019-2024) dibandingkan dengan **Prediksi Baru**.")
st.divider()

if st.sidebar.button("🔍 HITUNG & BANDINGKAN", type="primary"):
    input_data = np.array([[input_imunisasi, input_bblr, input_sanitasi]])
    hasil_prediksi = model.predict(input_data)[0]
    data_historis = df[df['nama_kabupaten_kota'] == pilih_kota]
    history_stunting = data_historis['jumlah_balita_stunting'].values
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Hasil Prediksi (Skenario Baru)", f"{int(hasil_prediksi):,} Jiwa")
    with col2:
        avg_history = history_stunting.mean()
        selisih = hasil_prediksi - avg_history
        st.metric(
            f"Rata-rata 5 Tahun Terakhir", 
            f"{int(avg_history):,} Jiwa",
            delta=f"{int(selisih)} (Selisih)",
            delta_color="inverse"
        ) 
    st.subheader(f"📊 Grafik: Tren 2019-2024 vs Prediksi")
    tahun_labels = ["2019", "2021", "2022", "2023", "2024"]
    labels = tahun_labels[:len(history_stunting)] + ["🔴 PREDIKSI BARU"]
    values = list(history_stunting) + [hasil_prediksi]
    colors = ['#cccccc'] * len(history_stunting) + ['#ff4b4b']
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors)
    ax.bar_label(bars, fmt='{:,.0f}', fontsize=10)
    ax.set_ylabel("Jumlah Balita Stunting")
    ax.set_title(f"Perbandingan Data Historis vs Simulasi di {pilih_kota}", fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)    
    st.pyplot(fig)
    st.info("ℹ️ **Data Abu-abu (2019-2024)** adalah data aktual dari dataset. **Data Merah** adalah hasil simulasi prediksi berdasarkan input Anda.")

else:
    st.info("👈 Silakan pilih kota dan masukkan data di sidebar untuk memulai analisis.")
