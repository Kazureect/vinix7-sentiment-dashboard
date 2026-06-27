import streamlit as st
import pandas as pd
import joblib
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

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
def load_models():
    # Menggunakan model LinearSVC (SVM) dan TF-IDF yang baru dibuat
    model = joblib.load('svm_sentiment_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer_dashboard.pkl')
    return model, tfidf

@st.cache_resource
def setup_nlp():
    # Inisialisasi Stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    # Memuat lexicon slang
    df_slang = pd.read_csv('colloquial-indonesian-lexicon.csv')
    slang_dict = dict(zip(df_slang['slang'], df_slang['formal']))
    
    # Menambahkan slang khusus domain WhatsApp/Aplikasi
    domain_specific_slang = {
        "vc": "panggilan video", "tc": "tes kontak", "sw": "status", 
        "wa": "whatsapp", "apk": "aplikasi", "gbs": "tidak bisa", 
        "lemot": "lambat", "bug": "eror"
    }
    slang_dict.update(domain_specific_slang)
    
    # Memuat Stopwords
    df_stop = pd.read_csv('stopwordbahasa.csv', header=None)
    kata_negasi = {'tidak', 'bukan', 'jangan', 'kurang', 'belum', 'enggak', 'ga', 'gak', 'tdk', 'g', 'nggak'}
    stopwords_set = set(df_stop[0].tolist())
    
    # PENTING: Menghapus kata negasi dari stopwords agar makna sentimen negatif terjaga
    stopwords_final = set(stopwords_set) - kata_negasi
    
    # Compile Regex untuk kecepatan (Dieksekusi satu kali di awal)
    clean_pattern = re.compile(r'[^a-zA-Z\s]')
    repeat_char_pattern = re.compile(r'(.)\1{2,}')
    
    return stemmer, slang_dict, stopwords_final, clean_pattern, repeat_char_pattern

# Global Cache Memory untuk Stemming
# Cache ini tidak akan terhapus selama server berjalan, membuat proses data baru instan!
@st.cache_resource
def get_stem_cache():
    return {}

# Load komponen ke memory
model, tfidf = load_models()
stemmer, slang_dict, stopwords_final, clean_pattern, repeat_char_pattern = setup_nlp()
stem_cache = get_stem_cache()

# ==========================================
# 3. FUNGSI PREPROCESSING SUPER CEPAT
# ==========================================
def preprocess_text(text):
    if pd.isna(text) or str(text).strip() == "":
        return ""
    
    # 1. Case folding & Regex Cleansing
    text = str(text).lower()
    text = repeat_char_pattern.sub(r'\1\1', text) # Menyingkat huruf berulang
    text = clean_pattern.sub(' ', text)           # Hapus karakter non-huruf
    text = re.sub(r'\s+', ' ', text).strip()      # Rapikan spasi
    
    # 2. Tokenization
    words = text.split()
    processed_words = []
    
    for word in words:
        # 3. Normalisasi Slang
        word = slang_dict.get(word, word)
        
        # 4. Stopwords Removal
        if word not in stopwords_final:
            # 5. Stemming dengan Teknik Memoization (Cache memory O(1))
            if word not in stem_cache:
                stem_cache[word] = stemmer.stem(word)
            processed_words.append(stem_cache[word])
            
    return ' '.join(processed_words)

# ==========================================
# 4. ANTARMUKA UPLOAD DATA
# ==========================================
uploaded_file = st.file_uploader("📂 Unggah file CSV Ulasan (Optimasi cepat hingga jutaan teks)", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"File berhasil diunggah! Total data: {len(df)} baris.")
    
    # Meminta user memilih kolom mana yang berisi teks ulasan
    kolom_teks = st.selectbox("Pilih kolom yang berisi teks ulasan aslinya:", df.columns)
    
    if st.button("🚀 Mulai Analisis"):
        with st.spinner("Sedang membersihkan teks dan melakukan prediksi (menggunakan cache turbo)..."):
            
            # Aplikasikan preprocessing super cepat
            df['teks_bersih'] = df[kolom_teks].apply(preprocess_text)
            
            # Hapus baris yang menjadi kosong akibat pre-processing
            df_valid = df[df['teks_bersih'] != ""].copy()
            
            # Prediksi menggunakan TF-IDF dan SVM (LinearSVC)
            X_vektor = tfidf.transform(df_valid['teks_bersih'])
            df_valid['Prediksi_Sentimen'] = model.predict(X_vektor)
            
        # ==========================================
        # 5. VISUALISASI BUSINESS REPORTING
        # ==========================================
        st.divider()
        st.subheader("📈 Hasil Analisis Sentimen (Model SVM)")
        
        col1, col2 = st.columns(2)
        
        # Visualisasi 1: Donut Chart Proporsi
        with col1:
            sentimen_count = df_valid['Prediksi_Sentimen'].value_counts()
            
            # Palet warna dinamis (Bisa menyesuaikan mode 2-kelas atau 3-kelas)
            color_map = {'Positif': '#66b3ff', 'Negatif': '#ff9999', 'Netral': '#ffcc99'}
            colors = [color_map.get(x, '#cccccc') for x in sentimen_count.index]
            
            fig1, ax1 = plt.subplots(figsize=(5, 5))
            ax1.pie(sentimen_count, labels=sentimen_count.index, autopct='%1.1f%%', 
                    colors=colors, startangle=90, wedgeprops=dict(width=0.4))
            ax1.set_title("Proporsi Sentimen Pelanggan")
            st.pyplot(fig1)
            
        # Visualisasi 2: Bar Chart Keluhan Utama
        with col2:
            df_negatif = df_valid[df_valid['Prediksi_Sentimen'] == 'Negatif']
            if not df_negatif.empty:
                teks_negatif = ' '.join(df_negatif['teks_bersih'].astype(str))
                # Menghilangkan kata yang tidak mengandung makna sentimen khusus (Stopword operasional dashboard)
                kata_abaikan = {'aplikasi', 'app', 'apk', 'whatsapp', 'wa', 'nya', 'sih', 'ya', 'dong', 'kasih', 'bikin', 'buat', 'pakai'}
                kata_negatif_bersih = [kata for kata in teks_negatif.split() if kata not in kata_abaikan]
                
                hitung_kata = Counter(kata_negatif_bersih)
                df_keluhan = pd.DataFrame(hitung_kata.most_common(10), columns=['Kata Kunci', 'Frekuensi'])
                
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                sns.barplot(x='Frekuensi', y='Kata Kunci', data=df_keluhan, palette='Reds_r', ax=ax2)
                ax2.set_title("Top 10 Fokus Keluhan Utama (Fitur Bermasalah)")
                st.pyplot(fig2)
            else:
                st.info("🎉 Hebat! Tidak ada keluhan (Sentimen Negatif) yang signifikan terdeteksi.")
        
        # ==========================================
        # 6. TABEL & EKSPOR DATA
        # ==========================================
        st.subheader("📝 Rincian Data Terprediksi")
        st.dataframe(df_valid[[kolom_teks, 'teks_bersih', 'Prediksi_Sentimen']]) 
        
        # Tombol Download
        csv_data = df_valid.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Unduh Hasil Lengkap Prediksi (CSV)",
            data=csv_data,
            file_name="hasil_prediksi_sentimen_vinix7.csv",
            mime="text/csv"
        )
