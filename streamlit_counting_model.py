import streamlit as st
import pandas as pd
import joblib

# Load TfidfVectorizer yang disimpan dengan joblib.dump()
vectorizer_file = 'vectorizer.pkl'

try:
    # Coba load vectorizer
    vectorizer = joblib.load(vectorizer_file)
    st.success("Vectorizer berhasil dimuat.")
    
    # Debug isi vectorizer
    if hasattr(vectorizer, 'idf_'):
        st.write("### Vectorizer sudah ditraining.")
        
        # Convert vocabulary dan IDF ke DataFrame
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
        st.warning("Vectorizer tidak memiliki atribut `idf_`. Mungkin belum ditraining.")
        st.write("#### Isi Vocabulary:")
        st.json(vectorizer.vocabulary_)
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat vectorizer: {str(e)}")
