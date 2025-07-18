import joblib
import numpy as np
import pandas as pd
import re
import unicodedata
from nltk.tokenize import sent_tokenize,word_tokenize
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack


tfidf_text = joblib.load("tfidf_text.joblib")
tfidf_keyword = joblib.load("tfidf_key.joblib")
tfidf_location = joblib.load("tfidf_loc.joblib")
binarizer = joblib.load("binarizer.joblib")
model = joblib.load("model.joblib")

def predict(location,keyword,text):
    location = location.lower()
    keyword = keyword.lower()
    text = text.lower()

    keyword_nan = 0 if keyword =='null' else 1
    location_nan = 0 if location =='null' else 1

    mojibake_freq = len(re.findall(r'[^\x00-\x7F]', text))  
    has_link = int(bool(re.search(r'(http[s]?://|www\.)', text, re.IGNORECASE)))

    text_hashtags = len(re.findall(r'#\w+', text))
    text_mentions = len(re.findall(r'@\w+', text))
    text_exclamations = text.count('!')
    text_questions = text.count('?')
    
    def is_suspicious_char(c):
        if ord(c) < 128:
            return c.isdigit()
        try:
            cat = unicodedata.category(c)
            return cat[0] not in {'L', 'Z', 'P'}
        except:
            return True

    def has_suspicious_location(text):
        if pd.isna(text):
            return 0
        for char in text:
            if is_suspicious_char(char):
                return 1
        if re.search(r"[^a-zA-Z\s'’\-éèêàçñöüïÉÈÊÀÇÑÖÜÏ]", text):
            return 1
        return 0
    
    location_spec = has_suspicious_location(location)

    text_char = len(text)
    text_words = len(word_tokenize(text))
    text_sent = len(sent_tokenize(text))

    def rep_score(text):
        words = text.lower().split()
        total_words = len(words)
        if total_words == 0:
            return 0.0
        word_counts = Counter(words)
        repeated = sum(count for count in word_counts.values() if count > 1)
        return repeated / total_words
    
    repetition_score = rep_score(text)

        
    def max_word_freq(text):
        words = text.lower().split()
        if not words:
            return 0.0
        word_counts = Counter(words)
        return max(word_counts.values()) / len(words)
    
    max_word_repeat_ratio = max_word_freq(text)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')) - {'no', 'not', 'nor', 'never'} 

    def transformtext(text):
        if pd.isna(text):
            return ""
        
        tokens = word_tokenize(text)
        
        clean_tokens = []
        for word in tokens:
            word_lower = word.lower()
            if (word_lower in stop_words):
                continue
            if re.match(r'^[a-zA-Z0-9]+$', word):  
                clean_tokens.append(lemmatizer.lemmatize(word_lower))
            elif re.match(r'^#\w+$', word):        
                clean_tokens.append(word_lower)
            elif re.match(r'^@\w+$', word):        
                clean_tokens.append(word_lower)
            elif word in {'!', '?'}:               
                clean_tokens.append(word)

        return " ".join(clean_tokens).strip()

    text = transformtext(text)

    features = ['keyword_nan', 'location_nan', 'mojibake_freq', 'has_link', 'text_hashtags', 'text_mentions', 'text_exclamations', 'text_questions', 'location_spec', 'text_char', 'text_words', 'text_sent']

    keyword_tfidf = tfidf_keyword.transform([keyword])
    location_tfidf = tfidf_location.transform([location])
    text_tfidf = tfidf_text.transform([text])

    tfidf_stack = hstack([keyword_tfidf, location_tfidf, text_tfidf])

    extra_features = np.array([
        keyword_nan, location_nan, mojibake_freq, has_link, text_hashtags,
        text_mentions, text_exclamations, text_questions, location_spec,
        text_char, text_words, text_sent
    ]).reshape(1, -1)

    X_final = hstack([tfidf_stack, extra_features])

    X_final_binarized = binarizer.transform(X_final)

    prediction = model.predict(X_final_binarized)[0]
    prob = model.predict_proba(X_final_binarized)[0]
    return prediction, prob

