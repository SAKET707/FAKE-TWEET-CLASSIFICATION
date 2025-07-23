🧪 FAKE TWEET CLASSIFIER

A machine learning-based classifier to detect potentially fake or misleading tweets using linguistic patterns, special tokens, and text analysis. This project leverages Bernoulli Naive Bayes with TF-IDF features and engineered metadata signals.
🌐 Streamlit App

URL: https://fake-tweet-classification-by-saket.streamlit.app/
🚀 Overview

This project builds a binary classification model to distinguish between real and fake tweets using both text and metadata. It achieves strong performance using a lightweight and interpretable model — ideal for baseline or production-ready moderation tools.
✅ Model Performance

    Model: Bernoulli Naive Bayes (BernoulliNB)

    Accuracy: 81.78%

    Classification Report:

                precision    recall  f1-score
        0       0.81        0.89    0.85
        1       0.83        0.73    0.78

    Class 0 = Real tweet, Class 1 = Fake tweet

📦 Feature Engineering
🔤 Textual Features (TF-IDF Vectorized)

    keyword — tweet-level keyword

    location — user-specified location

    text — main tweet content

    TF-IDF Vectorizer used for all three fields (with max 2000 features for text)

📊 Metadata Features

Engineered numeric features that capture patterns of tweet composition:

    keyword_nan — whether keyword is missing

    location_nan — whether location is missing

    mojibake_freq — presence of encoding glitches

    has_link — whether tweet includes a URL

    text_hashtags — count of hashtags

    text_mentions — count of user mentions

    text_exclamations — number of exclamation marks

    text_questions — number of question marks

    location_spec — whether location seems real/geocodable

    text_char — character count

    text_words — word count

    text_sent — sentence count

🧠 Model Choice

Bernoulli Naive Bayes was selected because:

    Features were binarized (0/1) using sklearn.preprocessing.Binarizer

    Suitable for high-dimensional sparse input (e.g., TF-IDF)

    Fast training and prediction

    Strong performance with textual and binary features

🛠️ Tech Stack

    Python 3.x

    scikit-learn

    pandas, numpy

    TfidfVectorizer for feature extraction

    BernoulliNB for classification
