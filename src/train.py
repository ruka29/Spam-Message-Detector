import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from joblib import dump
from src.preprocess import preprocess

def train_and_evaluate(train_path='data/spam.csv', test_path='data/testing_data.csv'):
    train_data = pd.read_csv(train_path, encoding='ISO-8859-1', usecols=[0, 1], names=["label", "message"], skiprows=1)
    X_train, vectorizer = preprocess(train_data['message'])
    y_train = train_data['label']
    y_train = y_train.map({'ham': 0, 'spam': 1})

    model = MultinomialNB()
    model.fit(X_train, y_train)

    dump(model, 'models/anomaly_model.pkl')
    dump(vectorizer, 'models/vectorizer.pkl')
    print("Model training complete.")

    test_data = pd.read_csv(test_path, encoding='ISO-8859-1', usecols=[0, 1], names=["label", "message"], skiprows=1)
    X_test = vectorizer.transform(test_data['message'])
    y_test = test_data['label']
    y_test = y_test.map({'ham': 0, 'spam': 1})

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy * 100, "%")
