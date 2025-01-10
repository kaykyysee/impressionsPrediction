import streamlit as st
import pandas as pd
import numpy as np
import scipy.sparse as sp
import joblib
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Load Model and Vectorizer
vectorizer_file = 'vectorizer.pkl'

# Load TfidfVectorizer yang disimpan dengan joblib.dump()
vectorizer = joblib.load(vectorizer_file)

# Streamlit Interface
st.title("Debugging Vectorizer")

# Debugging Vectorizer
if hasattr(vectorizer, 'idf_'):
    st.write("### Vectorizer sudah ditraining.")
    
    # Convert vocabulary and IDF to DataFrame for display
    vocab_data = pd.DataFrame({
        "Kata": list(vectorizer.vocabulary_.keys()),
        "Index": list(vectorizer.vocabulary_.values()),
        "IDF": [vectorizer.idf_[i] for i in vectorizer.vocabulary_.values()]
    }).sort_values(by="Index").reset_index(drop=True)
    
    st.write("#### Jumlah Vocabulary:", len(vectorizer.vocabulary_))
    st.write("#### Contoh Vocabulary dengan IDF:")
    st.dataframe(vocab_data.head(20))  # Tampilkan 20 kata pertama
    
    # Opsi untuk menampilkan seluruh vocabulary
    if st.checkbox("Tampilkan Seluruh Vocabulary"):
        st.dataframe(vocab_data)
else:
    st.error("Vectorizer belum ditraining. Silakan train vectorizer terlebih dahulu.")
