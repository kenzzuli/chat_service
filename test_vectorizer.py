from sklearn.feature_extraction.text import TfidfVectorizer
from dnn.recall.BM25vectorizer import BM25Vectorizer
from dnn.recall.fasttext_vectorizer import FastTextVectorizer

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
# model_by_char = get_model(by_char=True)
# model_by_word = get_model(by_char=False)

# print(fasttext_vectorizer(model_by_word, data[1], by_char=False))
# print(fasttext_vectorizer(model_by_char, data[1], by_char=True))
