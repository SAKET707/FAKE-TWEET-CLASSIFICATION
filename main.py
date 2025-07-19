import streamlit as st
from prediction_helper import predict

import nltk
from nltk.data import find
try:
    find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab') 
    
try:
    find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.tokenize import word_tokenize


st.title("Fake Tweet Classifier")

text = st.text_area("Enter the tweet text", "")
keyword = st.text_input("Enter the keyword (optional)", value="")
location = st.text_input("Enter the location (optional)", value="")

if st.button("Classify Tweet"):
    if not text.strip():
        st.warning("Tweet text cannot be empty.")
    else:
        keyword = keyword.strip() if keyword.strip() else "null"
        location = location.strip() if location.strip() else "null"

        pred,prob_array = predict(location,keyword,text)
        label = "🟢 Real" if pred == 0 else "🔴 Fake"
        
        st.success(f"Prediction: {label}")
        st.success(f"Probability: {prob_array}")
