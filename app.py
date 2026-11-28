import os
import re
import time
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
import streamlit as st
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from google import genai
from wordcloud import WordCloud

# ==========================================
# 1. KONFIGURASI HALAMAN DAN API KEY
# ==========================================
st.set_page_config(
    page_title="Vinix7 Sentiment Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Dashboard Analisis Sentimen Ulasan Pengguna")
st.write("Unggah data ulasan terbaru dari Play Store untuk memantau performa produk digital perusahaan.")
st.markdown(
    "Bisa gunakan [Dataset Contoh](https://drive.google.com/drive/folders/1m0HO-eF6Ie6RG3OvEZFKJQobYBYCfueG?usp=sharing) untuk mencoba dashboard."
)

# Inisialisasi Gemini Client menggunakan Streamlit Secrets
gemini_client = None
try:
    if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"]:
        gemini_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    else:
        st.sidebar.warning("API Key Gemini belum dikonfigurasi di Streamlit Secrets.")
except Exception:
    st.sidebar.warning("API Key Gemini belum dikonfigurasi di Streamlit Secrets.")

# ==========================================
# 2. VALIDASI ASET & CACHING MODEL / NLP
# ==========================================
REQUIRED_FILES = [
    "svm_sentiment_model.pkl",
    "tfidf_vectorizer_dashboard.pkl",
    "colloquial-indonesian-lexicon.csv",
    "stopwordbahasa.csv",
]

missing_files = [f for f in REQUIRED_FILES if not os.path.exists(f)]
if missing_files:
    st.error(
        f"File berikut tidak ditemukan di repositori: **{', '.join(missing_files)}**. "
        "Pastikan semua file model dan kamus berada di folder utama (root) GitHub."
    )
    st.stop()


@st.cache_resource
def load_models():
    model = joblib.load("svm_sentiment_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer_dashboard.pkl")

    known_root_words = set()
    if hasattr(tfidf, "transformer_list"):
        for name, trans in tfidf.transformer_list:
            # Compatibility shim jika dijalankan pada scikit-learn < 1.6
            if hasattr(trans, "_tfidf") and not hasattr(trans._tfidf, "_idf_diag"):
                idf = trans._tfidf.__dict__.get("idf_")
                if idf is not None:
                    trans._tfidf._idf_diag = sp.diags(
                        idf,
                        offsets=0,
                        shape=(len(idf), len(idf)),
                        format="csr",
                        dtype=idf.dtype,
                    )
            # Ekstrak 6.498 kata dasar unik dari kosakata word_tfidf
            if name == "word_tfidf" and hasattr(trans, "vocabulary_"):
                known_root_words = {
                    term for term in trans.vocabulary_.keys() if " " not in term
                }

    return model, tfidf, known_root_words


@st.cache_resource
def setup_nlp():
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    df_slang = pd.read_csv("colloquial-indonesian-lexicon.csv").dropna(subset=["slang", "formal"])
    slang_dict = dict(
        zip(
            df_slang["slang"].astype(str).str.strip().str.lower(),
            df_slang["formal"].astype(str).str.strip().str.lower(),
        )
    )

    # Kamus slang lintas domain (100% sinkron dengan Tahap 2 training)
    general_digital_slang = {
        # Umum dan Teknis Aplikasi
        "apk": "aplikasi", "app": "aplikasi", "apps": "aplikasi",
        "gbs": "tidak bisa", "gabisa": "tidak bisa", "gakbisa": "tidak bisa",
        "lemot": "lambat", "lelet": "lambat", "lola": "lambat",
        "bug": "eror", "ngebug": "eror", "error": "eror", "err": "eror",
        "fc": "keluar sendiri", "forceclose": "keluar sendiri",
        "lag": "lambat", "ngelag": "lambat", "ngelagg": "lambat",
        "min": "admin", "dev": "pengembang", "up": "perbarui", "update": "perbarui",
        "mantul": "mantap betul", "rekomen": "rekomendasi", "worth": "layak",

        # Komunikasi (WhatsApp)
        "vc": "panggilan video", "tc": "tes kontak", "sw": "status", "wa": "whatsapp",

        # Fintech & E-Commerce (DANA, Tokopedia, Gojek)
        "tf": "transfer", "trf": "transfer", "cs": "layanan pelanggan",
        "ongkir": "ongkos kirim", "co": "checkout", "verif": "verifikasi",
        "topup": "isi saldo", "acc": "setuju", "cancel": "batal",

        # Media Sosial dan Video Pendek (TikTok)
        "tt": "tiktok", "vt": "video tiktok", "vid": "video", "vidio": "video",
        "fyp": "rekomendasi", "komen": "komentar", "koment": "komentar",
        "dm": "pesan", "uplod": "unggah", "upload": "unggah", "post": "unggah",
        "ban": "blokir", "banned": "blokir", "keban": "blokir", "blok": "blokir",
        "buffering": "lambat", "bufer": "lambat", "live": "siaran langsung",
    }
    slang_dict.update(general_digital_slang)

    df_stop = pd.read_csv("stopwordbahasa.csv", header=None)
    stopwords_raw = set(df_stop[0].astype(str).str.strip().str.lower().tolist())

    # Pertahankan seluruh kata negasi (100% sinkron dengan Tahap 2 training)
    kata_negasi = {
        "tidak", "bukan", "jangan", "kurang", "belum", "enggak",
        "ga", "gak", "gk", "nggak", "ngga", "tdk", "tak", "kagak", "tanpa"
    }
    stop_words = stopwords_raw - kata_negasi

    # Stopwords khusus visualisasi agar kata terlalu umum tidak mendominasi grafik
    stopwords_visual = stopwords_raw | kata_negasi | {
        "aplikasi", "app", "apk", "nya", "sih", "ya", "dong", "aja", "saja",
        "yg", "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk"
    }

    # Regex pembersihan (100% sinkron dengan Tahap 2 & 3 training)
    url_pattern = re.compile(r"http\S+|www\.\S+")
    mention_pattern = re.compile(r"@\w+|#\w+")
    clean_pattern = re.compile(r"[^a-zA-Z\s]")
    repeat_char_pattern = re.compile(r"(.)\1{2,}")

    return (
        stemmer,
        slang_dict,
        stop_words,
        stopwords_visual,
        url_pattern,
        mention_pattern,
        clean_pattern,
        repeat_char_pattern,
    )


@st.cache_resource
def get_stem_cache():
    # Muat kamus pre-computed stem_dict.pkl jika tersedia di repositori
    if os.path.exists("stem_dict.pkl"):
        return dict(joblib.load("stem_dict.pkl"))
    return {}


model, tfidf, known_root_words = load_models()
(
    stemmer,
    slang_dict,
    stop_words,
    stopwords_visual,
    url_pattern,
    mention_pattern,
    clean_pattern,
    repeat_char_pattern,
) = setup_nlp()
stem_cache = get_stem_cache()
HAS_PRECOMPUTED_STEM = os.path.exists("stem_dict.pkl")


# ==========================================
# 3. FUNGSI PREPROCESSING & PEMBACAAN CSV
# ==========================================
def read_uploaded_csv(file_obj):
    """Membaca file CSV dengan fallback encoding dan separator otomatis."""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            file_obj.seek(0)
            df_try = pd.read_csv(file_obj, encoding=enc)
            if df_try.shape[1] == 1:
                file_obj.seek(0)
                df_semi = pd.read_csv(file_obj, encoding=enc, sep=";")
                if df_semi.shape[1] > 1:
                    return df_semi
            return df_try
        except Exception:
            continue
    file_obj.seek(0)
    return pd.read_csv(file_obj, errors="replace")


def normalize_text_tokens(text):
    """
    Membersihkan teks dan melakukan normalisasi slang per token
    (100% sinkron dengan dual_pipeline_processing di Tahap 3).
    """
    if pd.isna(text) or str(text).strip() == "":
        return [], []

    text = str(text).lower()
    text = url_pattern.sub(" ", text)
    text = mention_pattern.sub(" ", text)
    text = repeat_char_pattern.sub(r"\1", text)
    text = clean_pattern.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens_summarization = []
    tokens_cls_raw = []

    for word in text.split():
        word_normal = slang_dict.get(word, word)
        tokens_summarization.append(word_normal)
        for sub_word in word_normal.split():
            if len(sub_word) > 1 and sub_word not in stop_words:
                tokens_cls_raw.append(sub_word)

    return tokens_cls_raw, tokens_summarization


def preprocess_batch_fast(raw_texts):
    """
    Preprocessing jalur ganda berkecepatan tinggi:
    - Menggunakan stem_dict.pkl (O(1) lookup) + known_root_words dari vectorizer.
    - Jika stem_dict.pkl tersedia, 100% kata (min_freq=1) tetap di-stemming tanpa kompromi akurasi.
    """
    parsed_docs = [normalize_text_tokens(t) for t in raw_texts]

    token_counts = Counter(
        sub_word
        for cls_tokens, _ in parsed_docs
        for sub_word in cls_tokens
    )

    min_freq = 1 if (HAS_PRECOMPUTED_STEM or len(raw_texts) < 2000) else 3

    words_to_stem = [
        word
        for word, freq in token_counts.items()
        if freq >= min_freq
        and len(word) <= 25
        and word not in known_root_words
        and word not in stem_cache
    ]

    for word in words_to_stem:
        stem_cache[word] = stemmer.stem(word)

    teks_klasifikasi = []
    teks_summarization = []

    for cls_tokens, sum_tokens in parsed_docs:
        stemmed_tokens = [
            word if word in known_root_words else stem_cache.get(word, word)
            for word in cls_tokens
        ]
        teks_klasifikasi.append(" ".join(stemmed_tokens))
        teks_summarization.append(" ".join(sum_tokens))

    return teks_klasifikasi, teks_summarization


def detect_default_columns(columns):
    """Mendeteksi indeks kolom ulasan dan kolom tanggal secara otomatis jika tersedia."""
    cols_lower = [c.lower() for c in columns]

    text_candidates = ["content", "review", "ulasan", "komentar", "text", "teks", "comment"]
    default_text_idx = 0
    for cand in text_candidates:
        for idx, col_l in enumerate(cols_lower):
            if cand in col_l:
                default_text_idx = idx
                break
        else:
            continue
        break

    date_candidates = ["at", "date", "tanggal", "time", "waktu", "created"]
    default_date_idx = 0  # 0 adalah '-- Tidak Ada / Lewati --'
    for cand in date_candidates:
        for idx, col_l in enumerate(cols_lower):
            if cand == col_l or cand in col_l:
                default_date_idx = idx + 1
                break
        else:
            continue
        break

    return default_text_idx, default_date_idx


# ==========================================
# 4. ANTARMUKA UPLOAD DATA DAN PRATINJAU
# ==========================================
uploaded_file = st.file_uploader("Unggah file CSV Ulasan", type=["csv"])

if uploaded_file is not None:
    df = read_uploaded_csv(uploaded_file)
    st.success(f"File berhasil diunggah! Total data: **{len(df):,}** baris.")

    st.markdown("**Pratinjau Data (5 Baris Pertama):**")
    st.dataframe(df.head(), use_container_width=True)
    st.divider()

    default_text_idx, default_date_idx = detect_default_columns(df.columns)

    col_pilih1, col_pilih2 = st.columns(2)
    with col_pilih1:
        kolom_teks = st.selectbox(
            "Pilih kolom yang berisi teks ulasan/komentar (Wajib):",
            df.columns,
            index=default_text_idx,
        )
    with col_pilih2:
        opsi_tanggal = ["-- Tidak Ada / Lewati --"] + list(df.columns)
        kolom_tanggal = st.selectbox(
            "Pilih kolom tanggal untuk Time Series (Opsional):",
            opsi_tanggal,
            index=default_date_idx,
        )

    if st.button("Mulai Analisis", type="primary"):
        with st.spinner("Sedang memproses seluruh teks dan memprediksi sentimen..."):
            start_time = time.perf_counter()
            df_proc = df.copy()
            raw_series = df_proc[kolom_teks].fillna("").astype(str)

            unique_texts = raw_series.unique().tolist()
            u_cls, u_sum = preprocess_batch_fast(unique_texts)

            map_cls = dict(zip(unique_texts, u_cls))
            map_sum = dict(zip(unique_texts, u_sum))

            valid_unique = [
                (txt, cls) for txt, cls in zip(unique_texts, u_cls) if cls.strip() != ""
            ]

            if not valid_unique:
                st.warning(
                    "Tidak ditemukan teks ulasan yang valid setelah pembersihan. "
                    "Pastikan Anda memilih kolom teks ulasan yang benar."
                )
                st.session_state.pop("hasil_analisis", None)
            else:
                u_valid_raw, u_valid_cls = zip(*valid_unique)
                X_vektor = tfidf.transform(list(u_valid_cls))
                u_preds = model.predict(X_vektor)
                map_pred = dict(zip(u_valid_raw, u_preds))

                df_proc["teks_klasifikasi"] = raw_series.map(map_cls)
                df_proc["teks_summarization"] = raw_series.map(map_sum)
                df_proc["Prediksi_Sentimen"] = raw_series.map(map_pred)

                df_valid = df_proc[df_proc["Prediksi_Sentimen"].notna()].copy()
                elapsed_sec = time.perf_counter() - start_time

                st.session_state["hasil_analisis"] = {
                    "df_valid": df_valid,
                    "kolom_teks": kolom_teks,
                    "kolom_tanggal": kolom_tanggal,
                    "ai_summary": None,
                    "elapsed_sec": elapsed_sec,
                }

    # ==========================================
    # 5. BUSINESS REPORTING (PERSISTEN VIA SESSION STATE)
    # ==========================================
    if "hasil_analisis" in st.session_state:
        hasil = st.session_state["hasil_analisis"]
        df_valid = hasil["df_valid"]
        kolom_teks_aktif = hasil["kolom_teks"]
        kolom_tanggal_aktif = hasil["kolom_tanggal"]
        elapsed_sec = hasil.get("elapsed_sec")

        st.divider()
        st.subheader("Ringkasan Hasil Analisis Sentimen")
        if elapsed_sec is not None:
            st.caption(f"Selesai memproses {len(df_valid):,} ulasan valid dalam {elapsed_sec:.2f} detik.")

        total_valid = len(df_valid)
        counts = df_valid["Prediksi_Sentimen"].value_counts()
        jml_pos = int(counts.get("Positif", 0))
        jml_net = int(counts.get("Netral", 0))
        jml_neg = int(counts.get("Negatif", 0))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Ulasan Valid", f"{total_valid:,}")
        m2.metric("Positif", f"{jml_pos:,}", f"{(jml_pos / total_valid * 100):.1f}%")
        m3.metric("Netral", f"{jml_net:,}", f"{(jml_net / total_valid * 100):.1f}%")
        m4.metric("Negatif", f"{jml_neg:,}", f"{(jml_neg / total_valid * 100):.1f}%")

        st.write("")
        color_map = {"Positif": "#66b3ff", "Negatif": "#ff9999", "Netral": "#ffcc99"}
        col1, col2 = st.columns(2)

        with col1:
            sentimen_count = df_valid["Prediksi_Sentimen"].value_counts()
            colors = [color_map.get(x, "#cccccc") for x in sentimen_count.index]

            fig1, ax1 = plt.subplots(figsize=(6, 5))
            ax1.pie(
                sentimen_count,
                autopct="%1.1f%%",
                pctdistance=0.8,
                colors=colors,
                startangle=90,
                wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
                textprops={"fontsize": 9, "fontweight": "bold"},
            )
            ax1.set_title("Proporsi Sentimen Pelanggan", fontsize=12, fontweight="bold")
            ax1.legend(sentimen_count.index, loc="center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)

            fig1.tight_layout()
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)

        df_negatif = df_valid[df_valid["Prediksi_Sentimen"] == "Negatif"].copy()
        with col2:
            if not df_negatif.empty:
                teks_negatif = " ".join(df_negatif["teks_klasifikasi"].astype(str))
                kata_negatif_bersih = [
                    kata.lower()
                    for kata in teks_negatif.split()
                    if kata.lower() not in stopwords_visual and len(kata) > 2
                ]

                if kata_negatif_bersih:
                    hitung_kata = Counter(kata_negatif_bersih)
                    df_keluhan = pd.DataFrame(hitung_kata.most_common(10), columns=["Kata Kunci", "Frekuensi"])

                    fig2, ax2 = plt.subplots(figsize=(6, 5))
                    sns.barplot(
                        x="Frekuensi",
                        y="Kata Kunci",
                        hue="Kata Kunci",
                        data=df_keluhan,
                        palette="Reds_r",
                        legend=False,
                        ax=ax2,
                    )
                    ax2.set_title("Top 10 Fokus Keluhan Utama", fontsize=12, fontweight="bold")
                    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                    sns.despine(ax=ax2)

                    fig2.tight_layout()
                    st.pyplot(fig2, use_container_width=True)
                    plt.close(fig2)
                else:
                    st.info("Tidak cukup kata kunci keluhan untuk ditampilkan.")
            else:
                st.info("Tidak ada keluhan signifikan terdeteksi.")

        if kolom_tanggal_aktif != "-- Tidak Ada / Lewati --":
            st.divider()
            st.subheader("Tren Sentimen Ulasan per Bulan")

            df_ts = df_valid.copy()
            df_ts["_tanggal_parsed"] = pd.to_datetime(df_ts[kolom_tanggal_aktif], errors="coerce")
            df_ts = df_ts.dropna(subset=["_tanggal_parsed"])

            if not df_ts.empty:
                df_ts["Bulan"] = df_ts["_tanggal_parsed"].dt.to_period("M").astype(str)
                tren_bulanan = (
                    df_ts.groupby(["Bulan", "Prediksi_Sentimen"])
                    .size()
                    .unstack(fill_value=0)
                    .sort_index()
                )

                fig_ts, ax_ts = plt.subplots(figsize=(12, 4.5))
                line_color_map = {"Positif": "#3399ff", "Negatif": "#ff6666", "Netral": "#ffaa44"}

                for sentimen in ["Positif", "Negatif", "Netral"]:
                    if sentimen in tren_bulanan.columns:
                        ax_ts.plot(
                            tren_bulanan.index,
                            tren_bulanan[sentimen],
                            marker="o",
                            linewidth=2.2,
                            markersize=6,
                            label=sentimen,
                            color=line_color_map.get(sentimen, "#888888"),
                        )

                ax_ts.set_title("Perkembangan Jumlah Ulasan Berdasarkan Sentimen (Bulanan)", fontsize=12, pad=12)
                ax_ts.set_xlabel("Bulan")
                ax_ts.set_ylabel("Jumlah Ulasan")
                ax_ts.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                ax_ts.grid(axis="y", linestyle="--", alpha=0.4)
                ax_ts.legend(title="Sentimen", frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
                plt.setp(ax_ts.get_xticklabels(), rotation=45, ha="right")
                sns.despine(ax=ax_ts)

                fig_ts.tight_layout()
                st.pyplot(fig_ts, use_container_width=True)
                plt.close(fig_ts)
            else:
                st.warning(
                    f"Kolom '{kolom_tanggal_aktif}' tidak mengandung format tanggal yang valid, "
                    "sehingga grafik Time Series tidak dapat ditampilkan."
                )

        st.divider()
        st.subheader("Word Cloud Kata Kunci Ulasan")

        col_wc1, col_wc2 = st.columns(2)

        with col_wc1:
            df_positif = df_valid[df_valid["Prediksi_Sentimen"] == "Positif"]
            kata_positif = [
                kata.lower()
                for teks in df_positif["teks_klasifikasi"].astype(str)
                for kata in teks.split()
                if kata.lower() not in stopwords_visual and len(kata) > 2
            ]
            if kata_positif:
                wc_pos = WordCloud(
                    width=800,
                    height=450,
                    background_color="white",
                    colormap="Blues",
                    max_words=80,
                    collocations=False,
                ).generate(" ".join(kata_positif))

                fig_wc1, ax_wc1 = plt.subplots(figsize=(6, 4))
                ax_wc1.imshow(wc_pos, interpolation="bilinear")
                ax_wc1.axis("off")
                ax_wc1.set_title("Word Cloud Sentimen Positif", fontsize=11, pad=8)
                fig_wc1.tight_layout()
                st.pyplot(fig_wc1, use_container_width=True)
                plt.close(fig_wc1)
            else:
                st.info("Tidak cukup kata untuk menampilkan Word Cloud Positif.")

        with col_wc2:
            kata_negatif_wc = [
                kata.lower()
                for teks in df_negatif["teks_klasifikasi"].astype(str)
                for kata in teks.split()
                if kata.lower() not in stopwords_visual and len(kata) > 2
            ]
            if kata_negatif_wc:
                wc_neg = WordCloud(
                    width=800,
                    height=450,
                    background_color="white",
                    colormap="Reds",
                    max_words=80,
                    collocations=False,
                ).generate(" ".join(kata_negatif_wc))

                fig_wc2, ax_wc2 = plt.subplots(figsize=(6, 4))
                ax_wc2.imshow(wc_neg, interpolation="bilinear")
                ax_wc2.axis("off")
                ax_wc2.set_title("Word Cloud Sentimen Negatif", fontsize=11, pad=8)
                fig_wc2.tight_layout()
                st.pyplot(fig_wc2, use_container_width=True)
                plt.close(fig_wc2)
            else:
                st.info("Tidak cukup kata untuk menampilkan Word Cloud Negatif.")

        if not df_negatif.empty:
            st.divider()
            st.subheader("Rangkuman Keluhan Utama (AI Insight)")

            if gemini_client:
                if hasil["ai_summary"] is None:
                    with st.spinner("Gemini AI sedang membaca dan meringkas keluhan dari pelanggan..."):
                        df_negatif["panjang_teks"] = df_negatif["teks_summarization"].str.len()
                        keluhan_terpilih = df_negatif.sort_values(by="panjang_teks", ascending=False).head(50)
                        teks_untuk_diringkas = ".\n".join(keluhan_terpilih["teks_summarization"].tolist())

                        prompt = f"""
                        Anda adalah analis bisnis. Baca kumpulan keluhan pengguna aplikasi berikut ini.
                        Buatkan rangkuman eksekutif singkat (maksimal 3-4 kalimat) tentang masalah teknis 
                        atau fitur utama yang paling dikeluhkan pengguna. Gunakan bahasa Indonesia yang profesional.

                        Keluhan:
                        {teks_untuk_diringkas}
                        """

                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                response = gemini_client.models.generate_content(
                                    model="gemini-2.5-flash",
                                    contents=prompt,
                                )
                                hasil["ai_summary"] = response.text
                                break
                            except Exception as e:
                                if "503" in str(e) and attempt < max_retries - 1:
                                    time.sleep(2)
                                    continue
                                else:
                                    st.error(
                                        f"Gagal menghasilkan ringkasan AI setelah {max_retries} percobaan. "
                                        f"Server AI sedang sibuk. Pesan error: {e}"
                                    )
                                    break

                if hasil["ai_summary"]:
                    st.info(hasil["ai_summary"])
            else:
                st.warning("Silakan konfigurasikan GEMINI_API_KEY di Streamlit Secrets untuk mengaktifkan fitur ini.")

        st.divider()
        st.subheader("Rincian Data Terprediksi")
        kolom_tampil = [kolom_teks_aktif, "teks_klasifikasi", "teks_summarization", "Prediksi_Sentimen"]
        if kolom_tanggal_aktif != "-- Tidak Ada / Lewati --" and kolom_tanggal_aktif != kolom_teks_aktif:
            kolom_tampil.insert(0, kolom_tanggal_aktif)

        st.dataframe(df_valid[kolom_tampil], use_container_width=True)

        csv_data = df_valid.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="Unduh Hasil Lengkap Prediksi (CSV)",
            data=csv_data,
            file_name="hasil_prediksi_sentimen.csv",
            mime="text/csv",
        )
