"""
使用fasttext获取词向量
"""
from fasttext import FastText
import config
import numpy as np
import os


class FastTextVectorizer(object):
    def __init__(self):
        self.model = self.get_model()

    def get_model(self):
        """获取模型"""
        model_path = config.fasttext_vectorizer_path
        if os.path.exists(model_path):
            model = FastText.load_model(model_path)
        else:
            recall_corpus = config.recall_corpus
            model = FastText.train_unsupervised(input=recall_corpus, minCount=1, wordNgrams=2, epoch=20)
            model.save_model(model_path)
        return model

    def transform(self, data):
        """
        :param data: 句子组成的列表
        :return: 向量组成的列表
        """
        ret = list()
        for sentence in data:
            ret.append(self.model.get_sentence_vector(sentence))
        return np.array(ret)

    def fit_transform(self, data):
        """
        其实没有再次进行拟合，而是在实例化FasttextVectorize对象时已经拟合好了模型
        :param data: 句子组成的列表
        :return: 向量组成的列表
        """
        return self.transform(data)
