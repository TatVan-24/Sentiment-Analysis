from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

VECTORIZER_PATH = r"D:\merged_partition_content\Tu_Hoc\AIO2025\MyPoject\Module03\Sentiment Analysis\model\tfidf_vectorizer.pkl"
def vectorize(X_train, y_train, X_test):
    tfidf = TfidfVectorizer(max_features=10000)
    tfidf.fit(X_train, y_train)
    x_train_encoder = tfidf.transform(X_train)
    x_test_encoder = tfidf.transform(X_test)
    joblib.dump(tfidf, VECTORIZER_PATH)
    return x_train_encoder, x_test_encoder

def load_vectorize():
    return joblib.load(VECTORIZER_PATH)