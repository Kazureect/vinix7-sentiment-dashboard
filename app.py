import streamlit as st
import pandas as pd
import joblib
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Vinix7 Sentiment Dashboard", layout="wide")
st.title("📊 Dashboard Analisis Sentimen Ulasan Pengguna")
st.write("Unggah data ulasan terbaru dari Play Store untuk memantau performa produk digital perusahaan.")

# ==========================================
# 2. CACHING AGAR APLIKASI CEPAT
# ==========================================
@st.cache_resource
def load_models_and_assets():
    model = joblib.load('nb_model_biner.pkl')
    tfidf = joblib.load('tfidf_biner.pkl')
    
    # Memuat kamus stemming instan dari file JSON
    with open('kamus_stemming.json', 'r') as f:
        kamus_offline = json.load(f)
        
    return model, tfidf, kamus_offline

model, tfidf, kamus_offline = load_models_and_assets()

@st.cache_resource
def setup_nlp():
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    df_slang = pd.read_csv('colloquial-indonesian-lexicon.csv')
    slang_dict = dict(zip(df_slang['slang'], df_slang['formal']))
    
    df_stop = pd.read_csv('stopwordbahasa.csv', header=None)
    kata_negasi = ['tidak', 'bukan', 'belum', 'jangan', 'kurang', 'enggak', 'ga', 'gak', 'tdk', 'g', 'nggak']
    stopwords_tanpa_negasi = [k for k in df_stop[0].tolist() if k not in kata_negasi]
    domain_stopwords = ['whatsapp', 'wa', 'aplikasi', 'app', 'apk', 'bintang', 'kasih', 'ulasan', 'update', 'hp', 'telepon', 'nomor', 'versi', 'sih', 'nya', 'nih', 'oh', 'ya', 'kok', 'dong', 'deh', 'mah', 'tuh']
    stopwords_final = set(stopwords_tanpa_negasi + domain_stopwords)
    
    return stemmer, slang_dict, stopwords_final

model, tfidf = load_models()
stemmer, slang_dict, stopwords_final = setup_nlp()

# ==========================================
# 3. FUNGSI PREPROCESSING
# ==========================================
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Slang & Stopword
    words = text.split()
    words = [slang_dict.get(word, word) for word in words]
    words = [word for word in words if word not in stopwords_final]
    return ' '.join(words)

# ==========================================
# 4. ANTARMUKA UPLOAD DATA
# ==========================================
uploaded_file = st.file_uploader("📂 Unggah file CSV Ulasan", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File berhasil diunggah!")
    
    # Meminta user memilih kolom mana yang berisi teks ulasan
    kolom_teks = st.selectbox("Pilih kolom yang berisi teks ulasan:", df.columns)
    
if st.button("🚀 Mulai Analisis"):
        # Tambahkan progress bar agar UI lebih interaktif
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Sedang membersihkan teks dan menghapus noise...")
        df['teks_bersih'] = df[kolom_teks].apply(preprocess_text)
        progress_bar.progress(50)
        
        status_text.text("Melakukan stemming instan (Dictionary Lookup)...")
        # Keajaiban terjadi di sini! Tidak pakai Sastrawi lagi, hanya mencocokkan kata
        df['teks_stemmed'] = df['teks_bersih'].apply(
            lambda x: ' '.join([kamus_offline.get(kata, kata) for kata in x.split()])
        )
        progress_bar.progress(80)
        
        status_text.text("Menebak sentimen pengguna...")
        X_vektor = tfidf.transform(df['teks_stemmed'])
        df['Prediksi_Sentimen'] = model.predict(X_vektor)
        
        progress_bar.progress(100)
        status_text.text("Selesai dalam sekejap! 🎉")

        # ==========================================
        # 5. VISUALISASI BUSINESS REPORTING
        # ==========================================
        st.divider()
        st.subheader("📈 Hasil Analisis Sentimen")
        
        col1, col2 = st.columns(2)
        
        # Donut Chart Sentimen Keseluruhan
        with col1:
            sentimen_count = df['Prediksi_Sentimen'].value_counts()
            fig1, ax1 = plt.subplots(figsize=(5, 5))
            ax1.pie(sentimen_count, labels=sentimen_count.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], startangle=90, wedgeprops=dict(width=0.4))
            ax1.set_title("Proporsi Sentimen")
            st.pyplot(fig1)
            
        # Bar Chart Fokus Keluhan Utama
        with col2:
            df_negatif = df[df['Prediksi_Sentimen'] == 'Negatif']
            if not df_negatif.empty:
                teks_negatif = ' '.join(df_negatif['teks_stemmed'].astype(str))
                kata_abaikan = ['tidak', 'enggak', 'ga', 'gak', 'tdk', 'g', 'nggak', 'bukan', 'belum', 'jangan', 'kurang', 'tolong', 'mohon', 'bisa', 'kasih', 'bikin', 'buat', 'dapat', 'pakai', 'masuk', 'keluar']
                kata_negatif_bersih = [kata for kata in teks_negatif.split() if kata not in kata_abaikan]
                
                hitung_kata = Counter(kata_negatif_bersih)
                df_keluhan = pd.DataFrame(hitung_kata.most_common(10), columns=['Kata Kunci', 'Frekuensi'])
                
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                sns.barplot(x='Frekuensi', y='Kata Kunci', data=df_keluhan, palette='Reds_r', ax=ax2)
                ax2.set_title("Top 10 Fokus Keluhan (Sentimen Negatif)")
                st.pyplot(fig2)
            else:
                st.info("Hebat! Tidak ada keluhan (Sentimen Negatif) terdeteksi.")
        
        # Menampilkan Tabel Data
        st.subheader("📋 Rincian Data")
        st.dataframe(df[[kolom_teks, 'Prediksi_Sentimen']].head(100))
