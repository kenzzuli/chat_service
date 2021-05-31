from sklearn.feature_extraction.text import TfidfVectorizer
from dnn.recall.BM25vectorizer import BM25Vectorizer

data = [
    'hello world',
    'oh hello there',
    'Play it',
    'Play it again Sam',
]
tfidf_vec = TfidfVectorizer()
print(tfidf_vec.fit_transform(data).toarray())
bm25_vec = BM25Vectorizer()
print(bm25_vec.fit_transform(data))
