import streamlit as st
import numpy as np
import scipy.sparse as sp
import joblib
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Load Model and Vectorizer
model_file = 'lightgbm_model.pkl'
vectorizer_file = 'vectorizer.pkl'
scaler_file = 'scaler.pkl'

# Load TfidfVectorizer yang disimpan dengan joblib.dump()
vectorizer = joblib.load(vectorizer_file)

# Load Model dan Scaler
model = joblib.load(model_file)
scaler = joblib.load(scaler_file)

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

# Preprocessing Function
stemmer = StemmerFactory().create_stemmer()
stop_words_id = set(StopWordRemoverFactory().get_stop_words())
def preprocess_text(text):
    text = re.sub(r'http\S+|https\S+|www\S+|ftp\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    tokens = [token for token in tokens if token not in stop_words_id]
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)

# Streamlit Interface
st.title("Prediksi Impression Pembaca Berita Detikcom")

# Debugging Section: Display vectorizer attributes
st.subheader("Debugging Vectorizer")
if hasattr(vectorizer, 'idf_'):
    st.write("Vectorizer sudah ditraining.")
    st.write("Jumlah Vocabulary:", len(vectorizer.vocabulary_))
    st.write("Contoh IDF:")
    idf_data = {
        "Kata": list(vectorizer.vocabulary_.keys())[:10],
        "Index": list(vectorizer.vocabulary_.values())[:10],
        "IDF": [vectorizer.idf_[i] for i in list(vectorizer.vocabulary_.values())[:10]]
    }
    st.dataframe(pd.DataFrame(idf_data))
else:
    st.error("Vectorizer belum ditraining.")

# Input text
user_text = st.text_area("Masukkan Teks Artikel")
retweets = st.number_input("Masukkan Jumlah Retweets", min_value=0, value=0, step=1)
domain = st.selectbox("Pilih Domain", options=list(domain_mapping.keys()))

if st.button("Prediksi"):
    if user_text.strip():
        # Preprocess text
        processed_text = preprocess_text(user_text)
        
        # Periksa apakah vectorizer telah ditrain
        if not hasattr(vectorizer, 'idf_'):
            st.warning("Vectorizer belum ditrain. Silakan train vectorizer terlebih dahulu.")
        else:
            # TF-IDF transform
            text_vector = vectorizer.transform([processed_text])
            
            # Encode domain
            encoded_domain = sp.csr_matrix([[domain_mapping[domain]]])
            
            # Retweets sparse matrix
            retweets_sparse = sp.csr_matrix([[retweets]])
            
            # Combine features
            input_features = sp.hstack([text_vector, encoded_domain, retweets_sparse])
            
            scaled_features = scaler.transform(input_features)
            
            # Predict
            prediction = model.predict(scaled_features)
            
            # Display prediction
            st.success(f"Prediksi Impression: {prediction[0]:,.0f}")
    else:
        st.warning("Tolong masukkan teks untuk prediksi.")
