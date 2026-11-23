import streamlit as st
import pandas as pd
import joblib
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import time

# Import library baru untuk Gemini API
from google import genai

# ==========================================
# 1. KONFIGURASI HALAMAN DAN API KEY
# ==========================================
st.set_page_config(page_title="Vinix7 Sentiment Dashboard", layout="wide")
st.title("Dashboard Analisis Sentimen Ulasan Pengguna")
st.write("Unggah data ulasan terbaru dari Play Store untuk memantau performa produk digital perusahaan.")
st.markdown("Bisa gunakan [Dataset Contoh](https://drive.google.com/drive/folders/1m0HO-eF6Ie6RG3OvEZFKJQobYBYCfueG?usp=sharing) untuk mencoba dashboard.")

# Inisialisasi Gemini Client menggunakan Streamlit Secrets
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    gemini_client = None
    st.sidebar.warning("API Key Gemini belum dikonfigurasi di Streamlit Secrets.")

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
    stopwords_raw = set(df_stop[0].astype(str).str.strip().tolist())
    kata_negasi = {'tidak', 'bukan', 'jangan', 'kurang', 'belum', 'enggak', 'ga', 'gak', 'tdk', 'g', 'nggak'}
    stopwords_final = stopwords_raw - kata_negasi
    
    # Stopwords khusus visualisasi (termasuk kata negasi & kata terlalu umum agar grafik lebih informatif)
    stopwords_visual = stopwords_raw | kata_negasi | {
        'aplikasi', 'app', 'apk', 'nya', 'sih', 'ya', 'dong', 'aja', 'saja'
    }
    
    clean_pattern = re.compile(r'[^a-zA-Z\s]')
    repeat_char_pattern = re.compile(r'(.)\1{2,}')
    
    return stemmer, slang_dict, stopwords_final, stopwords_visual, clean_pattern, repeat_char_pattern

@st.cache_resource
def get_stem_cache():
    return {}

model, tfidf = load_models()
stemmer, slang_dict, stopwords_final, stopwords_visual, clean_pattern, repeat_char_pattern = setup_nlp()
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
        
        # Jalur SVM (Tanpa stopword)
        if word_normal not in stopwords_final:
            stemmed_word = stem_cache.get(word_normal, word_normal)
            tokens_klasifikasi.append(stemmed_word)
            
    return ' '.join(tokens_klasifikasi), ' '.join(tokens_summarization)

# ==========================================
# 4. ANTARMUKA UPLOAD DATA DAN PRATINJAU
# ==========================================
uploaded_file = st.file_uploader("Unggah file CSV Ulasan", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"File berhasil diunggah! Total data: {len(df)} baris.")
    
    st.markdown("**Pratinjau Data (5 Baris Pertama):**")
    st.dataframe(df.head())
    st.divider()
    
    # Pemilihan kolom ulasan (wajib) dan kolom tanggal (opsional)
    col_pilih1, col_pilih2 = st.columns(2)
    with col_pilih1:
        kolom_teks = st.selectbox(
            'Pilih kolom yang berisi teks ulasan/komentar (Wajib):',
            df.columns
        )
    with col_pilih2:
        opsi_tanggal = ['-- Tidak Ada / Lewati --'] + list(df.columns)
        kolom_tanggal = st.selectbox(
            'Pilih kolom tanggal untuk Time Series (Opsional):',
            opsi_tanggal
        )
    
    if st.button("Mulai Analisis"):
        with st.spinner("Sedang memproses seluruh teks dan memprediksi..."):
            
            # 1. Gabungkan seluruh teks dan bersihkan
            semua_teks_mentah = " ".join(df[kolom_teks].astype(str).tolist()).lower()
            semua_teks_bersih = clean_pattern.sub(' ', semua_teks_mentah)
            
            # 2. Hitung frekuensi kata menggunakan Counter
            hitung_semua_kata = Counter(semua_teks_bersih.split())
            
            # 3. STRATEGI PEMANGKASAN: Hanya ambil kata yang muncul >= 3 kali
            kata_penting = {kata for kata, freq in hitung_semua_kata.items() if freq >= 3}
            
            # 4. Pre-populate Cache HANYA untuk kata penting
            for kata in kata_penting:
                kata_normal = slang_dict.get(kata, kata)
                if kata_normal not in stopwords_final and kata_normal not in stem_cache:
                    stem_cache[kata_normal] = stemmer.stem(kata_normal)
                    
            # 5. Eksekusi pemrosesan kilat dengan List Comprehension
            teks_mentah_list = df[kolom_teks].astype(str).tolist()
            hasil_proses = [dual_pipeline_process(teks) for teks in teks_mentah_list]
            
            # Pecah hasil kembali ke dataframe
            df['teks_klasifikasi'] = [res[0] for res in hasil_proses]
            df['teks_summarization'] = [res[1] for res in hasil_proses]
            
            df_valid = df[df['teks_klasifikasi'] != ""].copy()
            
            # Prediksi SVM
            X_vektor = tfidf.transform(df_valid['teks_klasifikasi'])
            df_valid['Prediksi_Sentimen'] = model.predict(X_vektor)
            
        # ==========================================
        # 5. BUSINESS REPORTING
        # ==========================================
        st.divider()
        st.subheader("Hasil Analisis Sentimen")
        
        color_map = {'Positif': '#66b3ff', 'Negatif': '#ff9999', 'Netral': '#ffcc99'}
        col1, col2 = st.columns(2)

        # Visualisasi 1: Proporsi Sentimen Pelanggan
        with col1:
            sentimen_count = df_valid['Prediksi_Sentimen'].value_counts()
            colors = [color_map.get(x, '#cccccc') for x in sentimen_count.index]
            
            fig1, ax1 = plt.subplots(figsize=(6, 5)) 
            ax1.pie(sentimen_count, 
                    autopct='%1.0f%%',  
                    pctdistance=0.8,    
                    colors=colors, 
                    startangle=90, 
                    wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
                    textprops={'fontsize': 8, 'fontweight': 'bold'})
            ax1.set_title("Proporsi Sentimen Pelanggan")
            ax1.legend(sentimen_count.index, loc="center", bbox_to_anchor=(0.5, -0.1), ncol=3)
            
            fig1.tight_layout()
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)
            
        # Visualisasi 2: Bar Chart Keluhan Utama
        with col2:
            df_negatif = df_valid[df_valid['Prediksi_Sentimen'] == 'Negatif'].copy()
            if not df_negatif.empty:
                teks_negatif = ' '.join(df_negatif['teks_klasifikasi'].astype(str))
                kata_negatif_bersih = [
                    kata.lower() for kata in teks_negatif.split() 
                    if kata.lower() not in stopwords_visual
                ]
                
                if kata_negatif_bersih:
                    hitung_kata = Counter(kata_negatif_bersih)
                    df_keluhan = pd.DataFrame(hitung_kata.most_common(10), columns=['Kata Kunci', 'Frekuensi'])
                    
                    fig2, ax2 = plt.subplots(figsize=(6, 5))
                    sns.barplot(x='Frekuensi', y='Kata Kunci', data=df_keluhan, palette='Reds_r', ax=ax2)
                    ax2.set_title("Top 10 Fokus Keluhan Utama")
                    sns.despine(ax=ax2)
                    
                    fig2.tight_layout()
                    st.pyplot(fig2, use_container_width=True)
                    plt.close(fig2)
                else:
                    st.info("Tidak cukup kata kunci keluhan untuk ditampilkan.")
            else:
                st.info("🎉 Tidak ada keluhan signifikan terdeteksi.")

        # ==========================================
        # 5B. TIME SERIES (PER BULAN)
        # ==========================================
        if kolom_tanggal != '-- Tidak Ada / Lewati --':
            st.divider()
            st.subheader("Tren Sentimen Ulasan per Bulan")
            
            df_ts = df_valid.copy()
            df_ts['_tanggal_parsed'] = pd.to_datetime(df_ts[kolom_tanggal], errors='coerce')
            df_ts = df_ts.dropna(subset=['_tanggal_parsed'])
            
            if not df_ts.empty:
                df_ts['Bulan'] = df_ts['_tanggal_parsed'].dt.to_period('M').astype(str)
                tren_bulanan = (
                    df_ts.groupby(['Bulan', 'Prediksi_Sentimen'])
                    .size()
                    .unstack(fill_value=0)
                    .sort_index()
                )
                
                fig_ts, ax_ts = plt.subplots(figsize=(12, 4.5))
                
                # Warna garis sedikit lebih tegas namun tetap senada dengan pie chart
                line_color_map = {'Positif': '#3399ff', 'Negatif': '#ff6666', 'Netral': '#ffaa44'}
                
                for sentimen in ['Positif', 'Negatif', 'Netral']:
                    if sentimen in tren_bulanan.columns:
                        ax_ts.plot(
                            tren_bulanan.index,
                            tren_bulanan[sentimen],
                            marker='o',
                            linewidth=2.2,
                            markersize=6,
                            label=sentimen,
                            color=line_color_map.get(sentimen, '#888888')
                        )
                
                ax_ts.set_title("Perkembangan Jumlah Ulasan Berdasarkan Sentimen (Bulanan)", fontsize=12, pad=12)
                ax_ts.set_xlabel("Bulan")
                ax_ts.set_ylabel("Jumlah Ulasan")
                ax_ts.grid(axis='y', linestyle='--', alpha=0.4)
                ax_ts.legend(title="Sentimen", frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
                plt.setp(ax_ts.get_xticklabels(), rotation=45, ha='right')
                sns.despine(ax=ax_ts)
                
                fig_ts.tight_layout()
                st.pyplot(fig_ts, use_container_width=True)
                plt.close(fig_ts)
            else:
                st.warning(f"Kolom '{kolom_tanggal}' tidak mengandung format tanggal yang valid, sehingga grafik Time Series tidak dapat ditampilkan.")

        # ==========================================
        # 5C. WORD CLOUD
        # ==========================================
        st.divider()
        st.subheader("Word Cloud Kata Kunci Ulasan")
        
        col_wc1, col_wc2 = st.columns(2)
        
        # Word Cloud Sentimen Positif
        with col_wc1:
            df_positif = df_valid[df_valid['Prediksi_Sentimen'] == 'Positif']
            kata_positif = [
                kata.lower()
                for teks in df_positif['teks_klasifikasi'].astype(str)
                for kata in teks.split()
                if kata.lower() not in stopwords_visual
            ]
            if kata_positif:
                wc_pos = WordCloud(
                    width=800,
                    height=450,
                    background_color='white',
                    colormap='Blues',
                    max_words=80,
                    collocations=False
                ).generate(' '.join(kata_positif))
                
                fig_wc1, ax_wc1 = plt.subplots(figsize=(6, 4))
                ax_wc1.imshow(wc_pos, interpolation='bilinear')
                ax_wc1.axis('off')
                ax_wc1.set_title("Word Cloud Sentimen Positif", fontsize=11, pad=8)
                fig_wc1.tight_layout()
                st.pyplot(fig_wc1, use_container_width=True)
                plt.close(fig_wc1)
            else:
                st.info("Tidak cukup kata untuk menampilkan Word Cloud Positif.")

        # Word Cloud Sentimen Negatif
        with col_wc2:
            kata_negatif_wc = [
                kata.lower()
                for teks in df_negatif['teks_klasifikasi'].astype(str)
                for kata in teks.split()
                if kata.lower() not in stopwords_visual
            ]
            if kata_negatif_wc:
                wc_neg = WordCloud(
                    width=800,
                    height=450,
                    background_color='white',
                    colormap='Reds',
                    max_words=80,
                    collocations=False
                ).generate(' '.join(kata_negatif_wc))
                
                fig_wc2, ax_wc2 = plt.subplots(figsize=(6, 4))
                ax_wc2.imshow(wc_neg, interpolation='bilinear')
                ax_wc2.axis('off')
                ax_wc2.set_title("Word Cloud Sentimen Negatif", fontsize=11, pad=8)
                fig_wc2.tight_layout()
                st.pyplot(fig_wc2, use_container_width=True)
                plt.close(fig_wc2)
            else:
                st.info("Tidak cukup kata untuk menampilkan Word Cloud Negatif.")
                
        # ==========================================
        # 6. SUMMARIZATION (AI INSIGHT)
        # ==========================================
        if not df_negatif.empty:
            st.divider()
            st.subheader("Rangkuman Keluhan Utama (AI Insight)")
            
            if gemini_client:
                with st.spinner("Gemini AI sedang membaca dan meringkas keluhan dari pelanggan..."):
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
                    
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            response = gemini_client.models.generate_content(
                                model='gemini-2.5-flash',
                                contents=prompt
                            )
                            st.info(response.text)
                            break 
                            
                        except Exception as e:
                            if "503" in str(e) and attempt < max_retries - 1:
                                time.sleep(2) 
                                continue
                            else:
                                st.error(f"Gagal menghasilkan ringkasan AI setelah {max_retries} percobaan. Server AI sedang sibuk. Pesan error: {e}")
                                break
            else:
                st.warning("Silakan konfigurasikan GEMINI_API_KEY di Streamlit Secrets untuk mengaktifkan fitur ini.")

        # ==========================================
        # 7. TABEL DAN EKSPOR DATA
        # ==========================================
        st.divider()
        st.subheader("Rincian Data Terprediksi")
        kolom_tampil = [kolom_teks, 'teks_klasifikasi', 'teks_summarization', 'Prediksi_Sentimen']
        if kolom_tanggal != '-- Tidak Ada / Lewati --' and kolom_tanggal != kolom_teks:
            kolom_tampil.insert(0, kolom_tanggal)
            
        st.dataframe(df_valid[kolom_tampil]) 
        
        csv_data = df_valid.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Unduh Hasil Lengkap Prediksi (CSV)",
            data=csv_data,
            file_name="hasil_prediksi_sentimen.csv",
            mime="text/csv"
        )
