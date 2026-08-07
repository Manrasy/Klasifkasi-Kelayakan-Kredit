import streamlit as st
import pandas as pd
import joblib
import requests
import os


# ==================================================
# KONFIGURASI MODEL
# ==================================================

RF_URL = (
    "https://huggingface.co/spaces/RSYD21/Kelayakan_Kredit/"
    "resolve/main/random_forest.pkl"
)

ENCODER_URL = (
    "https://huggingface.co/spaces/RSYD21/Kelayakan_Kredit/"
    "resolve/main/encoder.pkl"
)

SCALER_URL = (
    "https://huggingface.co/spaces/RSYD21/Kelayakan_Kredit/"
    "resolve/main/scaler.pkl"
)


RF_PATH = "random_forest.pkl"
ENCODER_PATH = "encoder.pkl"
SCALER_PATH = "scaler.pkl"



# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_models():

    files = [
        (RF_URL, RF_PATH),
        (ENCODER_URL, ENCODER_PATH),
        (SCALER_URL, SCALER_PATH)
    ]


    for url, path in files:

        if not os.path.exists(path):

            response = requests.get(url)

            if response.status_code == 200:

                with open(path, "wb") as file:
                    file.write(response.content)

            else:

                st.error(
                    f"Gagal mengambil file {path}"
                )

                st.stop()



    rf = joblib.load(RF_PATH)

    encoders = joblib.load(ENCODER_PATH)

    scaler = joblib.load(SCALER_PATH)


    return rf, encoders, scaler



rf, encoders, scaler = load_models()



# ==================================================
# KONFIGURASI HALAMAN
# ==================================================

st.set_page_config(
    page_title="Prediksi Kelayakan Kredit",
    page_icon="💳",
    layout="wide"
)



# ==================================================
# HEADER
# ==================================================

st.title(
    "💳 Sistem Klasifikasi Kelayakan Kredit"
)


st.write(
"""
Aplikasi ini menggunakan algoritma **Random Forest**
untuk melakukan klasifikasi kelayakan kredit
berdasarkan karakteristik calon nasabah.
"""
)


st.divider()



# ==================================================
# INPUT DATA NASABAH
# ==================================================

st.subheader(
    "📌 Input Data Calon Nasabah"
)


col1, col2, col3 = st.columns(3)



with col1:


    usia = st.number_input(
        "Usia",
        min_value=18,
        max_value=100,
        value=30
    )


    status_pekerjaan = st.selectbox(
        "Status Pekerjaan",
        [
            "Tetap",
            "Kontrak",
            "Tidak Bekerja",
            "Wirausaha"
        ]
    )


    lama_bekerja_tahun = st.number_input(
        "Lama Bekerja (Tahun)",
        min_value=0,
        value=5
    )


    pendapatan_tahunan = st.number_input(
        "Pendapatan Tahunan",
        min_value=0,
        value=60000000
    )


    skor_kredit = st.number_input(
        "Skor Kredit",
        min_value=0,
        max_value=900,
        value=700
    )


    lama_riwayat_kredit_tahun = st.number_input(
        "Lama Riwayat Kredit (Tahun)",
        min_value=0,
        value=5
    )



with col2:


    aset_tabungan = st.number_input(
        "Aset Tabungan",
        min_value=0,
        value=10000000
    )


    hutang_saat_ini = st.number_input(
        "Hutang Saat Ini",
        min_value=0,
        value=5000000
    )


    gagal_bayar_tercatat = st.number_input(
        "Gagal Bayar Tercatat",
        min_value=0,
        value=0
    )


    tunggakan_2thn_terakhir = st.number_input(
        "Tunggakan 2 Tahun Terakhir",
        min_value=0,
        value=0
    )


    catatan_negatif = st.number_input(
        "Catatan Negatif",
        min_value=0,
        value=0
    )


    tipe_produk = st.selectbox(
        "Tipe Produk",
        [
            "Personal",
            "Konsumtif",
            "Bisnis"
        ]
    )



with col3:


    tujuan_pinjaman = st.selectbox(
        "Tujuan Pinjaman",
        [
            "Modal Usaha",
            "Pendidikan",
            "Rumah",
            "Konsumsi"
        ]
    )


    jumlah_pinjaman = st.number_input(
        "Jumlah Pinjaman",
        min_value=0,
        value=20000000
    )


    suku_bunga = st.number_input(
        "Suku Bunga (%)",
        min_value=0.0,
        value=10.0
    )


    rasio_hutang_terhadap_pendapatan = st.number_input(
        "Rasio Hutang terhadap Pendapatan",
        min_value=0.0,
        value=0.3
    )


    rasio_pinjaman_terhadap_pendapatan = st.number_input(
        "Rasio Pinjaman terhadap Pendapatan",
        min_value=0.0,
        value=0.5
    )


    rasio_pembayaran_terhadap_pendapatan = st.number_input(
        "Rasio Pembayaran terhadap Pendapatan",
        min_value=0.0,
        value=0.2
    )



st.divider()



# ==================================================
# PREDIKSI
# ==================================================

if st.button(
    "🔍 Prediksi Kelayakan Kredit",
    use_container_width=True
):


    data_input = pd.DataFrame({

        "usia":[usia],

        "status_pekerjaan":[status_pekerjaan],

        "lama_bekerja_tahun":[lama_bekerja_tahun],

        "pendapatan_tahunan":[pendapatan_tahunan],

        "skor_kredit":[skor_kredit],

        "lama_riwayat_kredit_tahun":[lama_riwayat_kredit_tahun],

        "aset_tabungan":[aset_tabungan],

        "hutang_saat_ini":[hutang_saat_ini],

        "gagal_bayar_tercatat":[gagal_bayar_tercatat],

        "tunggakan_2thn_terakhir":[tunggakan_2thn_terakhir],

        "catatan_negatif":[catatan_negatif],

        "tipe_produk":[tipe_produk],

        "tujuan_pinjaman":[tujuan_pinjaman],

        "jumlah_pinjaman":[jumlah_pinjaman],

        "suku_bunga":[suku_bunga],

        "rasio_hutang_terhadap_pendapatan":[
            rasio_hutang_terhadap_pendapatan
        ],

        "rasio_pinjaman_terhadap_pendapatan":[
            rasio_pinjaman_terhadap_pendapatan
        ],

        "rasio_pembayaran_terhadap_pendapatan":[
            rasio_pembayaran_terhadap_pendapatan
        ]

    })



    try:


        # ==========================================
        # ENCODING
        # ==========================================

        for col in data_input.select_dtypes(
            include="object"
        ).columns:


            data_input[col] = encoders[col].transform(
                data_input[col]
            )



        # ==========================================
        # SCALING
        # ==========================================

        data_scaled = scaler.transform(
            data_input
        )



        # ==========================================
        # RANDOM FOREST PREDICTION
        # ==========================================

        prediction = rf.predict(
            data_scaled
        )


        probability = rf.predict_proba(
            data_scaled
        )



        hasil = prediction[0]


        confidence = (
            max(probability[0]) * 100
        )



        # ==========================================
        # OUTPUT
        # ==========================================

        st.subheader(
            "📊 Hasil Prediksi"
        )


        if hasil == 1:

            st.success(
                "✅ Nasabah Layak Kredit"
            )

        else:

            st.error(
                "❌ Nasabah Tidak Layak Kredit"
            )



        st.metric(
            "Tingkat Keyakinan Model",
            f"{confidence:.2f}%"
        )



        st.subheader(
            "Detail Input"
        )


        st.dataframe(
            data_input,
            use_container_width=True
        )



    except Exception as e:


        st.error(
            f"Terjadi kesalahan saat prediksi: {e}"
        )
