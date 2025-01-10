import streamlit as st
import numpy as np
import joblib
import re
from sentence_transformers import SentenceTransformer
import scipy.sparse as sp

# ==========================
# 1. Load Model dan BERT
# ==========================
model_file = 'lightgbm_model.pkl'
scaler_file = 'scaler.pkl'

model = joblib.load(model_file)
scaler = joblib.load(scaler_file)

# Load pre-trained BERT model
bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Domain mapping
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
# 2. Preprocessing Function
# ==========================
def preprocess_text(text):
    text = re.sub(r'http\S+|https\S+|www\S+|ftp\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

# ==========================
# 3. Streamlit Interface
# ==========================
st.title("Prediksi Impression Pembaca Postingan Berita Detik.com")

# Input text
user_text = st.text_area("Masukkan Teks Postingan X")
retweets = st.number_input("Masukkan Jumlah Retweets", min_value=0, value=0, step=1)
domain = st.selectbox("Pilih Domain", options=list(domain_mapping.keys()))

if st.button("Prediksi"):
    if user_text.strip():
        # Preprocess text
        processed_text = preprocess_text(user_text)
        text_length = len(processed_text)

        # Convert text to BERT embeddings
        text_embedding = bert_model.encode([processed_text])
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
