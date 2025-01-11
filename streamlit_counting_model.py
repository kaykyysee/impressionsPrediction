import streamlit as st
import numpy as np
import joblib
import re
from sentence_transformers import SentenceTransformer
import scipy.sparse as sp
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import string

# Unduh dataset punkt jika belum tersedia
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ==========================
# 1. Fungsi Caching untuk Model dan SentenceTransformer
# ==========================
@st.cache_resource
def load_model_and_bert():
    model_file = 'lightgbm_model.pkl'
    scaler_file = 'scaler.pkl'

    # Load LightGBM model dan scaler
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)

    # Load pre-trained BERT model
    bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    return model, scaler, bert_model

# Load sekali saja model dan SentenceTransformer
model, scaler, bert_model = load_model_and_bert()

# ==========================
# 2. Domain Mapping
# ==========================
domain_mapping = {
    'news.detik.com': 7,
    'detik.com': 0,
    'hot.detik.com': 5,
    'wolipop.detik.com': 11,
    'health.detik.com': 4,
    'finance.detik.com': 1,
    'sport.detik.com': 9,
    'inet.detik.com': 6,
    'food.detik.com': 2,
    'travel.detik.com': 10,
    'oto.detik.com': 8,
    'haibunda.com': 3,
}

# ==========================
# 3. Fungsi Preprocessing dan Pembersihan Teks
# ==========================
@st.cache_data
def preprocess_text(text):
    """
    Preprocessing teks untuk menghapus URL, tanda baca, dan huruf besar.
    """
    # Hapus URL
    text = re.sub(r'http\S+|https\S+|www\S+|ftp\S+', '', text)  # Hapus URL
    text = re.sub(r'\b[a-zA-Z0-9]+\.com\S*', '', text)  # Hapus domain seperti example.com/link
    text = re.sub(r'\b[a-zA-Z0-9]+detikcom\S*', '', text)  # Hapus detikcom yang berubah format

    # Hapus tanda baca dan ubah teks menjadi huruf kecil
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()

    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()

    # Hapus spasi berlebih setelah tagar
    text = re.sub(r'#(\s+)', '#', text)

    return text

@st.cache_data
def clean_text_id(text):
    """
    Fungsi untuk membersihkan teks, menghapus stop words, dan melakukan stemming.
    """
    # Tokenisasi dan pembersihan
    tokens = word_tokenize(text)

    # Inisialisasi stop words dan stemmer
    stop_words_id = set(StopWordRemoverFactory().get_stop_words())
    stemmer = StemmerFactory().create_stemmer()

    # Hapus stop words dan lakukan stemming
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words_id]

    return ' '.join(tokens)

@st.cache_data
def get_text_embedding(processed_text):
    """
    Menghasilkan embedding teks menggunakan SentenceTransformer.
    """
    return bert_model.encode([processed_text])

# ==========================
# 4. Streamlit Interface
# ==========================
st.title("Prediksi Impression Pembaca Postingan Berita Detik.com")

# Input text
user_text = st.text_area("Masukkan Teks Postingan X")
retweets = st.number_input("Masukkan Jumlah Retweets", min_value=0, value=0, step=1)
domain = st.selectbox("Pilih Domain", options=list(domain_mapping.keys()))

if st.button("Prediksi"):
    if user_text.strip():
        # Preprocess text
        st.write("Melakukan preprocessing teks...")
        processed_text = preprocess_text(user_text)
        cleaned_text = clean_text_id(processed_text)
        text_length = len(cleaned_text)

        # Convert text to BERT embeddings
        st.write("Menghasilkan embedding...")
        text_embedding = get_text_embedding(cleaned_text)
        text_sparse = sp.csr_matrix(text_embedding)

        # Encode domain
        encoded_domain = sp.csr_matrix([[domain_mapping[domain]]])

        # Retweets sparse matrix
        retweets_sparse = sp.csr_matrix([[retweets]])

        # Length sparse matrix
        length_sparse = sp.csr_matrix([[text_length]])

        # Combine features
        input_features = sp.hstack([text_sparse, encoded_domain, retweets_sparse, length_sparse])
        scaled_features = scaler.transform(input_features)

        # Predict
        prediction = model.predict(scaled_features)
        st.success(f"Perkiraan pembaca sebanyak: {prediction[0]:,.0f} orang")
    else:
        st.warning("Tolong masukkan teks untuk prediksi.")
