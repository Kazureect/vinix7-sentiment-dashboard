import streamlit as st
import pandas as pd
import joblib
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Import library baru untuk Gemini API
from google import genai

# ==========================================
# 1. KONFIGURASI HALAMAN & API KEY
# ==========================================
st.set_page_config(page_title="Vinix7 Sentiment Dashboard", layout="wide")
st.title("📊 Dashboard Analisis Sentimen Ulasan Pengguna")
st.write("Unggah data ulasan terbaru dari Play Store untuk memantau performa produk digital perusahaan.")

# Inisialisasi Gemini Client menggunakan Streamlit Secrets
try:
    # Mengambil API Key dari brankas rahasia Streamlit
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    gemini_client = None
    st.sidebar.warning("⚠️ API Key Gemini belum dikonfigurasi di Streamlit Secrets.")

# ==========================================
# 2. CACHING AGAR APLIKASI CEPAT
# ==========================================
@st.cache_resource
def load_models():
    model = joblib.load('svm_sentiment_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer_dashboard.pkl')
    return model, tfidf

@st.cache_resource
def setup_nlp():
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    df_slang = pd.read_csv('colloquial-indonesian-lexicon.csv')
    slang_dict = dict(zip(df_slang['slang'], df_slang['formal']))
    
    domain_specific_slang = {
        "vc": "panggilan video", "tc": "tes kontak", "sw": "status", 
        "wa": "whatsapp", "apk": "aplikasi", "gbs": "tidak bisa", 
        "lemot": "lambat", "bug": "eror"
    }
    slang_dict.update(domain_specific_slang)
    
    df_stop = pd.read_csv('stopwordbahasa.csv', header=None)
    kata_negasi = {'tidak', 'bukan', 'jangan', 'kurang', 'belum', 'enggak', 'ga', 'gak', 'tdk', 'g', 'nggak'}
    stopwords_final = set(df_stop[0].tolist()) - kata_negasi
    
    clean_pattern = re.compile(r'[^a-zA-Z\s]')
    repeat_char_pattern = re.compile(r'(.)\1{2,}')
    
    return stemmer, slang_dict, stopwords_final, clean_pattern, repeat_char_pattern

@st.cache_resource
def get_stem_cache():
    return {}

model, tfidf = load_models()
stemmer, slang_dict, stopwords_final, clean_pattern, repeat_char_pattern = setup_nlp()
stem_cache = get_stem_cache()

# ==========================================
# 3. FUNGSI PREPROCESSING JALUR GANDA
# ==========================================
def dual_pipeline_process(text):
    if pd.isna(text) or str(text).strip() == "":
        return "", ""
    
    text = str(text).lower()
    text = repeat_char_pattern.sub(r'\1\1', text) 
    text = clean_pattern.sub(' ', text)           
    text = re.sub(r'\s+', ' ', text).strip()      
    
    words = text.split()
    tokens_klasifikasi = []
    tokens_summarization = []
    
    for word in words:
        word_normal = slang_dict.get(word, word)
        
        # Jalur LLM (Tetap utuh)
        tokens_summarization.append(word_normal)
        
        # Jalur SVM (Tanpa stopword & di-stemming)
        if word_normal not in stopwords_final:
            if word_normal not in stem_cache:
                stem_cache[word_normal] = stemmer.stem(word_normal)
            tokens_klasifikasi.append(stem_cache[word_normal])
            
    return ' '.join(tokens_klasifikasi), ' '.join(tokens_summarization)

# ==========================================
# 4. ANTARMUKA UPLOAD DATA
# ==========================================
uploaded_file = st.file_uploader("📂 Unggah file CSV Ulasan", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"File berhasil diunggah! Total data: {len(df)} baris.")
    
    kolom_teks = st.selectbox("Pilih kolom yang berisi teks ulasan aslinya:", df.columns)
    
    if st.button("🚀 Mulai Analisis"):
        with st.spinner("Sedang memproses teks (Dual Pipeline) dan melakukan prediksi..."):
            
            # Eksekusi fungsi jalur ganda secara efisien
            hasil_proses = df[kolom_teks].apply(dual_pipeline_process)
            df['teks_klasifikasi'] = [res[0] for res in hasil_proses]
            df['teks_summarization'] = [res[1] for res in hasil_proses]
            
            df_valid = df[df['teks_klasifikasi'] != ""].copy()
            
            # Prediksi SVM menggunakan kolom teks_klasifikasi
            X_vektor = tfidf.transform(df_valid['teks_klasifikasi'])
            df_valid['Prediksi_Sentimen'] = model.predict(X_vektor)
            
        # ==========================================
        # 5. VISUALISASI BUSINESS REPORTING
        # ==========================================
        st.divider()
        st.subheader("📈 Hasil Analisis Sentimen")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sentimen_count = df_valid['Prediksi_Sentimen'].value_counts()
            color_map = {'Positif': '#66b3ff', 'Negatif': '#ff9999', 'Netral': '#ffcc99'}
            colors = [color_map.get(x, '#cccccc') for x in sentimen_count.index]
            
            fig1, ax1 = plt.subplots(figsize=(5, 5))
            ax1.pie(sentimen_count, labels=sentimen_count.index, autopct='%1.1f%%', 
                    colors=colors, startangle=90, wedgeprops=dict(width=0.4))
            ax1.set_title("Proporsi Sentimen Pelanggan")
            st.pyplot(fig1)
            
        with col2:
            df_negatif = df_valid[df_valid['Prediksi_Sentimen'] == 'Negatif'].copy()
            if not df_negatif.empty:
                teks_negatif = ' '.join(df_negatif['teks_klasifikasi'].astype(str))
                kata_abaikan = {'aplikasi', 'app', 'apk', 'whatsapp', 'wa', 'nya', 'sih', 'ya', 'dong', 'kasih', 'bikin', 'buat', 'pakai'}
                kata_negatif_bersih = [kata for kata in teks_negatif.split() if kata not in kata_abaikan]
                
                hitung_kata = Counter(kata_negatif_bersih)
                df_keluhan = pd.DataFrame(hitung_kata.most_common(10), columns=['Kata Kunci', 'Frekuensi'])
                
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                sns.barplot(x='Frekuensi', y='Kata Kunci', data=df_keluhan, palette='Reds_r', ax=ax2)
                ax2.set_title("Top 10 Fokus Keluhan Utama")
                st.pyplot(fig2)
            else:
                st.info("🎉 Tidak ada keluhan signifikan terdeteksi.")
                
        # ==========================================
        # 6. SUMMARIZATION (INSIGHT AI)
        # ==========================================
        if not df_negatif.empty:
            st.divider()
            st.subheader("💡 Rangkuman Keluhan Utama (AI Insight)")
            
            if gemini_client:
                with st.spinner("Gemini AI sedang membaca dan meringkas keluhan dari pelanggan..."):
                    # Ambil 50 keluhan terpanjang dari teks_summarization (teks utuh)
                    df_negatif['panjang_teks'] = df_negatif['teks_summarization'].apply(len)
                    keluhan_terpilih = df_negatif.sort_values(by='panjang_teks', ascending=False).head(50)
                    teks_untuk_diringkas = ".\n".join(keluhan_terpilih['teks_summarization'].tolist())
                    
                    prompt = f"""
                    Anda adalah analis bisnis. Baca kumpulan keluhan pengguna aplikasi berikut ini.
                    Buatkan rangkuman eksekutif singkat (maksimal 3-4 kalimat) tentang masalah teknis 
                    atau fitur utama yang paling dikeluhkan pengguna. Gunakan bahasa Indonesia yang profesional.

                    Keluhan:
                    {teks_untuk_diringkas}
                    """
                    try:
                        response = gemini_client.models.generate_content(
                            model='gemini-1.5-flash',
                            contents=prompt
                        )
                        st.info(response.text)
                    except Exception as e:
                        st.error(f"Gagal menghasilkan ringkasan AI: {e}")
            else:
                st.warning("Silakan konfigurasikan GEMINI_API_KEY di Streamlit Secrets untuk mengaktifkan fitur ini.")

        # ==========================================
        # 7. TABEL & EKSPOR DATA
        # ==========================================
        st.subheader("📝 Rincian Data Terprediksi")
        st.dataframe(df_valid[[kolom_teks, 'teks_klasifikasi', 'teks_summarization', 'Prediksi_Sentimen']]) 
        
        csv_data = df_valid.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Unduh Hasil Lengkap Prediksi (CSV)",
            data=csv_data,
            file_name="hasil_prediksi_sentimen_vinix7.csv",
            mime="text/csv"
        )