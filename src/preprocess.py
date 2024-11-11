import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = list(stopwords.words('english'))  # Convert the set to a list

def preprocess(texts):
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=1000)
    text_vectors = vectorizer.fit_transform(texts)
    return text_vectors, vectorizer
