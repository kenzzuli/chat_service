import json
import os
import config
from sklearn.feature_extraction.text import TfidfVectorizer
from dnn.recall.BM25vectorizer import BM25Vectorizer
import pysparnn.cluster_index as ci
import pickle
from config import by_char


class Sentence2Vector(object):
    def __init__(self, vectorize_method="BM25"):
        self.qa_dict = json.load(open(config.qa_path, "r"))
        self.vectorize_method = vectorize_method

    def build_all(self):
        """获取文字、向量、tfidf转换器、索引"""
        key = "q_cut_by_char" if by_char else "q_cut_by_word"
        lines = [q for q in self.qa_dict.keys()]
        lines_cut = [" ".join(self.qa_dict[q][key]) for q in lines]
        vectorizer = BM25Vectorizer() if self.vectorize_method.lower == "bm25" else TfidfVectorizer()
        features_vec = vectorizer.fit_transform(lines_cut)
        search_index = self.get_index(features_vec, lines)
        return vectorizer, features_vec, lines_cut, search_index

    def get_index(self, vectors, data):
        """获取索引，如果不存在则构建"""
        if os.path.exists(config.search_index_path):
            return pickle.load(open(config.search_index_path, "rb"))
        else:
            return self.build_index(vectors, data)

    def build_index(self, vectors, data):
        """构建索引，并存储"""
        search_index = ci.MultiClusterIndex(vectors, data, num_indexes=2)
        pickle.dump(search_index, open(config.search_index_path, "wb"))
        return search_index
