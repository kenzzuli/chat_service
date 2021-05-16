"""
意图识别的封装
"""
import fasttext
import config
from lib.cut_sentence import cut


class Classify(object):
    def __init__(self):
        """
        加载训练好的模型
        """
        # 词语特征模型
        self.by_word_model = fasttext.load_model(config.classify_model_by_word_path)
        # 单字特征模型
        self.by_char_model = fasttext.load_model(config.classify_model_by_char_path)

    def predict(self, sentence):
        """
        预测输入数据的结果
        :param sentence 句子
        :return (label, acc)
        """
        # 预测结果的格式为((label1,), array([置信度1,])
        result_by_word = self.by_word_model.predict(" ".join(cut(sentence, by_character=False)).strip())
        result_by_char = self.by_char_model.predict(" ".join(cut(sentence, by_character=True)).strip())
        # *号对元组进行解包，获取元组中的每个元素，得到 (label1,), array([置信度1,]), (label2,),  array([置信度2,]) 
        # 此时，zip中有四个可迭代对象,分别是 元组, ndarray, 元组, ndarray
        # zip会取每个可迭代对象的第1个元素组成第1个元组，每个可迭代对象的第2个元素组成第2个元组，以元组为元素，组成新的zip可迭代对象
        # 使用list将zip对象变成列表 即 [ (label1, 置信度1, label2, 置信度2), ]
        # 取列表的第0个元素 即 (label1, 置信度1, label2, 置信度2)
        label_by_word, confidence_by_word, label_by_char, confidence_by_char = \
        list(zip(*result_by_word, *result_by_char))[0]

        # 把所有的置信度，都转到同一个标签比较大小
        if label_by_word == "__label__chat":
            confidence_by_word = 1 - confidence_by_word

        if label_by_char == "__label__chat":
            confidence_by_char = 1 - confidence_by_char

        # 设置阈值
        threshold = 0.9
        if confidence_by_word > threshold or confidence_by_char > threshold:
            return "QA", max(confidence_by_word, confidence_by_char)  # 返回两者中最大的置信值
        else:  # 由于置信度已经转成了类别为QA的置信度，所以如果类别是chat，要把置信度再转回来
            return "chat", 1 - min(confidence_by_word, confidence_by_char)

        # # TODO 假如有三个类别
        # threshold = 0.9
        # if label_by_word == label_by_char:  # 如果两个模型预测结果相同
        #     # 如果二者中有一个满足阈值要求
        #     if confidence_by_word > threshold or confidence_by_char > threshold:
        #         return label_by_word, max(confidence_by_word, confidence_by_char)
        #     else:
        #         return None, 0  # 无法判断分类
        # else:  # 如果不同， 先以char为准，因为模型评估时发现，char比word准确点
        #     if confidence_by_char > threshold:  # 先看by_char的结果,如果大于某个值，直接返回
        #         return label_by_char, confidence_by_char
        #     elif confidence_by_word > threshold:  # 再看by_word的结果，如果大于某个值，直接返回
        #         return label_by_word, confidence_by_word
        #     else:  # 如果都不满足
        #         return None, 0  # 无法判断分类
