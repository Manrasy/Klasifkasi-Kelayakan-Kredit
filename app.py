from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==================================================
# KONFIGURASI HALAMAN (harus perintah Streamlit pertama)
# ==================================================
st.set_page_config(
    page_title="Prediksi Kelayakan Kredit",
    page_icon="💳",
    layout="wide",
)

# ==================================================
# KONFIGURASI MODEL (file .pkl ada di repo, satu folder dengan app.py)
# ==================================================
BASE_DIR = Path(__file__).parent

RF_PATH = BASE_DIR / "random_forest.pkl"
ENCODER_PATH = BASE_DIR / "encoder.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

# Urutan kolom bawaan (dipakai jika model tidak menyimpan nama kolom)
DEFAULT_FEATURES = [
    "usia",
    "status_pekerjaan",
    "lama_bekerja_tahun",
    "pendapatan_tahunan",
    "skor_kredit",
    "lama_riwayat_kredit_tahun",
    "aset_tabungan",
    "hutang_saat_ini",
    "gagal_bayar_tercatat",
    "tunggakan_2thn_terakhir",
    "catatan_negatif",
    "tipe_produk",
    "tujuan_pinjaman",
    "jumlah_pinjaman",
    "suku_bunga",
    "rasio_hutang_terhadap_pendapatan",
    "rasio_pinjaman_terhadap_pendapatan",
    "rasio_pembayaran_terhadap_pendapatan",
]

CATEGORICAL = ["status_pekerjaan", "tipe_produk", "tujuan_pinjaman"]

DEFAULT_OPTIONS = {
    "status_pekerjaan": ["Tetap", "Kontrak", "Tidak Bekerja", "Wirausaha"],
    "tipe_produk": ["Personal", "Konsumtif", "Bisnis"],
    "tujuan_pinjaman": ["Modal Usaha", "Pendidikan", "Rumah", "Konsumsi"],
}


# ==================================================
# FUNGSI BANTU
# ==================================================
def norm(s):
    """Normalisasi nama: huruf kecil, tanpa spasi/underscore/strip."""
    return str(s).lower().replace(" ", "").replace("_", "").replace("-", "")


@st.cache_resource(show_spinner="Memuat model...")
def load_models():
    for path in (RF_PATH, ENCODER_PATH, SCALER_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"File {path.name} tidak ada di repo. "
                f"Upload ke folder yang sama dengan app.py."
            )

    rf = joblib.load(RF_PATH)
    encoders = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    return rf, encoders, scaler


def get_encoder(encoders, col):
    """Ambil encoder untuk satu kolom (jika berupa dict)."""
    if isinstance(encoders, dict):
        peta = {norm(k): k for k in encoders}
        kunci = peta.get(norm(col))
        return encoders[kunci] if kunci is not None else None
    return None


def get_options(encoders, col):
    """Pilihan selectbox: ikuti label saat model dilatih jika tersedia."""
    enc = get_encoder(encoders, col)
    classes = getattr(enc, "classes_", None)
    if classes is not None:
        return [str(c) for c in classes]

    # OrdinalEncoder tunggal: categories_ berurutan sesuai feature_names_in_
    names = getattr(encoders, "feature_names_in_", None)
    cats = getattr(encoders, "categories_", None)
    if names is not None and cats is not None:
        for n, c in zip(names, cats):
            if norm(n) == norm(col):
                return [str(x) for x in c]

    return DEFAULT_OPTIONS[col]


def encode_value(encoders, col, df):
    """Ubah satu kolom kategori menjadi angka."""
    enc = get_encoder(encoders, col)

    if enc is not None:
        classes = getattr(enc, "classes_", None)
        nilai = df[col].iloc[0]
        if classes is not None and nilai not in list(classes):
            raise ValueError(
                f"Nilai '{nilai}' tidak dikenal untuk '{col}'. "
                f"Pilihan valid: {list(classes)}"
            )
        try:
            return enc.transform(df[col])
        except Exception:
            return np.ravel(enc.transform(df[[col]]))

    # Encoder tunggal (mis. OrdinalEncoder untuk banyak kolom)
    names = getattr(encoders, "feature_names_in_", None)
    if names is not None and hasattr(encoders, "transform"):
        cols_enc = list(names)
        sub = pd.DataFrame(
            {n: [df[next(c for c in df.columns if norm(c) == norm(n))].iloc[0]]
             for n in cols_enc}
        )
        hasil = encoders.transform(sub)
        idx = [norm(n) for n in cols_enc].index(norm(col))
        return hasil[:, idx]

    # encoder.pkl kosong: tebakan terbaik = LabelEncoder bawaan (urut abjad)
    opsi = sorted(DEFAULT_OPTIONS[next(c for c in DEFAULT_OPTIONS if norm(c) == norm(col))])
    nilai = df[col].iloc[0]
    st.session_state["fallback_encoding"] = True
    return [opsi.index(nilai)]


def feature_order(rf, scaler):
    for obj in (scaler, rf):
        names = getattr(obj, "feature_names_in_", None)
        if names is not None:
            return list(names)
    return DEFAULT_FEATURES


# ==================================================
# LOAD MODEL
# ==================================================
try:
    rf, encoders, scaler = load_models()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()


# ==================================================
# HEADER
# ==================================================
st.title("💳 Sistem Klasifikasi Kelayakan Kredit")
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
st.subheader("📌 Input Data Calon Nasabah")

col1, col2, col3 = st.columns(3)

with col1:
    usia = st.number_input("Usia", min_value=18, max_value=100, value=30)
    status_pekerjaan = st.selectbox(
        "Status Pekerjaan", get_options(encoders, "status_pekerjaan")
    )
    lama_bekerja_tahun = st.number_input(
        "Lama Bekerja (Tahun)", min_value=0, value=5
    )
    pendapatan_tahunan = st.number_input(
        "Pendapatan Tahunan", min_value=0, value=60_000_000, step=1_000_000
    )
    skor_kredit = st.number_input(
        "Skor Kredit", min_value=0, max_value=900, value=700
    )
    lama_riwayat_kredit_tahun = st.number_input(
        "Lama Riwayat Kredit (Tahun)", min_value=0, value=5
    )

with col2:
    aset_tabungan = st.number_input(
        "Aset Tabungan", min_value=0, value=10_000_000, step=1_000_000
    )
    hutang_saat_ini = st.number_input(
        "Hutang Saat Ini", min_value=0, value=5_000_000, step=1_000_000
    )
    gagal_bayar_tercatat = st.number_input(
        "Gagal Bayar Tercatat", min_value=0, value=0
    )
    tunggakan_2thn_terakhir = st.number_input(
        "Tunggakan 2 Tahun Terakhir", min_value=0, value=0
    )
    catatan_negatif = st.number_input("Catatan Negatif", min_value=0, value=0)
    tipe_produk = st.selectbox(
        "Tipe Produk", get_options(encoders, "tipe_produk")
    )

with col3:
    tujuan_pinjaman = st.selectbox(
        "Tujuan Pinjaman", get_options(encoders, "tujuan_pinjaman")
    )
    jumlah_pinjaman = st.number_input(
        "Jumlah Pinjaman", min_value=0, value=20_000_000, step=1_000_000
    )
    suku_bunga = st.number_input("Suku Bunga (%)", min_value=0.0, value=10.0)
    rasio_hutang_terhadap_pendapatan = st.number_input(
        "Rasio Hutang terhadap Pendapatan", min_value=0.0, value=0.3
    )
    rasio_pinjaman_terhadap_pendapatan = st.number_input(
        "Rasio Pinjaman terhadap Pendapatan", min_value=0.0, value=0.5
    )
    rasio_pembayaran_terhadap_pendapatan = st.number_input(
        "Rasio Pembayaran terhadap Pendapatan", min_value=0.0, value=0.2
    )

st.divider()


# ==================================================
# PREDIKSI
# ==================================================
if st.button("🔍 Prediksi Kelayakan Kredit", use_container_width=True):

    nilai_input = {
        "usia": usia,
        "status_pekerjaan": status_pekerjaan,
        "lama_bekerja_tahun": lama_bekerja_tahun,
        "pendapatan_tahunan": pendapatan_tahunan,
        "skor_kredit": skor_kredit,
        "lama_riwayat_kredit_tahun": lama_riwayat_kredit_tahun,
        "aset_tabungan": aset_tabungan,
        "hutang_saat_ini": hutang_saat_ini,
        "gagal_bayar_tercatat": gagal_bayar_tercatat,
        "tunggakan_2thn_terakhir": tunggakan_2thn_terakhir,
        "catatan_negatif": catatan_negatif,
        "tipe_produk": tipe_produk,
        "tujuan_pinjaman": tujuan_pinjaman,
        "jumlah_pinjaman": jumlah_pinjaman,
        "suku_bunga": suku_bunga,
        "rasio_hutang_terhadap_pendapatan": rasio_hutang_terhadap_pendapatan,
        "rasio_pinjaman_terhadap_pendapatan": rasio_pinjaman_terhadap_pendapatan,
        "rasio_pembayaran_terhadap_pendapatan": rasio_pembayaran_terhadap_pendapatan,
    }

    try:
        # Susun kolom sesuai urutan saat model dilatih
        peta_input = {norm(k): k for k in nilai_input}
        urutan = feature_order(rf, scaler)

        baris = {}
        st.session_state["fallback_encoding"] = False

        for nama in urutan:
            kunci = peta_input.get(norm(nama))
            if kunci is None:
                # Kolom hasil one-hot, mis. "status_pekerjaan_Tetap"
                for cat in CATEGORICAL:
                    if str(nama).lower().startswith(cat.lower() + "_"):
                        kategori = str(nama)[len(cat) + 1:]
                        baris[nama] = [
                            1.0 if norm(nilai_input[cat]) == norm(kategori) else 0.0
                        ]
                        break
                if nama in baris:
                    continue
                raise KeyError(
                    f"Fitur '{nama}' dari model tidak ada di form. "
                    f"Fitur form: {list(nilai_input.keys())}"
                )
            baris[nama] = [nilai_input[kunci]]

        data_input = pd.DataFrame(baris)
        data_tampil = data_input.copy()

        # Encoding kolom kategori
        for col in data_input.columns:
            if norm(col) in [norm(c) for c in CATEGORICAL]:
                data_input[col] = encode_value(encoders, col, data_input)

        data_input = data_input.astype(float)

        if st.session_state.get("fallback_encoding"):
            st.warning(
                "encoder.pkl kosong, jadi kolom kategori di-encode urut abjad "
                "(tebakan). Pastikan sama dengan saat training."
            )

        # Scaling
        if getattr(scaler, "feature_names_in_", None) is not None:
            data_scaled = scaler.transform(data_input)
        else:
            data_scaled = scaler.transform(data_input.values)

        # Prediksi
        if getattr(rf, "feature_names_in_", None) is not None:
            data_scaled = pd.DataFrame(data_scaled, columns=data_input.columns)

        prediksi = rf.predict(data_scaled)[0]
        proba = rf.predict_proba(data_scaled)[0]
        kelas = list(rf.classes_)

        layak = str(prediksi).lower() in ("1", "layak", "true", "ya", "yes")
        keyakinan = float(np.max(proba)) * 100

        # Output
        st.subheader("📊 Hasil Prediksi")

        if layak:
            st.success("✅ Nasabah Layak Kredit")
        else:
            st.error("❌ Nasabah Tidak Layak Kredit")

        m1, m2 = st.columns(2)
        m1.metric("Tingkat Keyakinan Model", f"{keyakinan:.2f}%")
        m2.metric("Kelas Prediksi", str(prediksi))

        with st.expander("Probabilitas per kelas"):
            st.dataframe(
                pd.DataFrame(
                    {"Kelas": [str(k) for k in kelas],
                     "Probabilitas (%)": [round(p * 100, 2) for p in proba]}
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Detail Input")
        st.dataframe(data_tampil, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
        with st.expander("Info diagnostik"):
            st.write("Tipe encoders:", str(type(encoders)))
            if isinstance(encoders, dict):
                st.write("Kunci encoders:", list(encoders.keys()))
            fs = getattr(scaler, "feature_names_in_", None)
            fr = getattr(rf, "feature_names_in_", None)
            st.write("Fitur scaler:", [] if fs is None else list(fs))
            st.write("Fitur RF:", [] if fr is None else list(fr))
            st.write("Jumlah fitur scaler:", getattr(scaler, "n_features_in_", None))
