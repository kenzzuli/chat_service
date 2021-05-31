"""
返回召回的结果
"""
from dnn.recall.sentence_vector import Sentence2Vector
from lib import cut
from config import by_char
import config


class Recall(object):
    def __init__(self, vectorize_method="fasttext"):
        self.s2v = Sentence2Vector(vectorize_method=vectorize_method)
        self.vectorizer, self.features_vec, self.lines_cut, self.search_index = self.s2v.build_all()

    def predict(self, sentence: str):
        """
        输入问题，返回最相近的问题
        :param sentence: 要搜索的问题
        :return:
        """
        sentence_cut = [" ".join(cut(sentence, by_character=by_char))]
        # ['python 真的 很 简单 吗 ？', '什么 是 产品经理 ？'] 以空格作为分隔
        search_vector = self.vectorizer.transform(sentence_cut)
        search_results = self.search_index.search(search_vector, k=config.recall_nums,
                                                  k_clusters=config.recall_clusters,
                                                  num_indexes=2,
                                                  return_distance=True)
        # [[('0.0', '蒋夏梦是谁？'), ('1.0', 'python真的很简单吗？'), ('1.0', '什么是产品经理？'), ('1.0', '什么样的人适合做产品经理呀？')]]
        final_result = list()
        # 过滤实体entity
        # 获取用户输入的问题中的实体
        sentence_cut_with_pos = cut(sentence, by_character=False, with_pos=True)
        q_entity = [i[0] for i in sentence_cut_with_pos if i[1] == "kc" or i[1] == "shisu"]
        # 判断是否存在相同实体
        for i in search_results:
            for j in i:
                matched_q = j[1]
                matched_q_entity = self.s2v.qa_dict[matched_q]["entity"]
                if len(set(matched_q_entity) & set(q_entity)) > 0:  # 集合取交集
                    final_result.append(matched_q)
        # 如果存在相同实体，则返回匹配到的存在相同实体的问题
        if len(final_result) > 0:
            return final_result
        # 如果不存在相同实体，则返回原始结果
        else:
            return [j[1] for j in i for i in search_results]
