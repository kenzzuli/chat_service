"""
句子转序列
"""
from tqdm import tqdm
import pickle


class Sen2Seq(object):
    UNK_TAG = "UNK"  # 未知
    PAD_TAG = "PAD"  # 填充
    SOS_TAG = "SOS"  # 句子开始
    EOS_TAG = "EOS"  # 句子结束

    def __init__(self):
        self.dict = {self.UNK_TAG: 0, self.PAD_TAG: 1, self.SOS_TAG: 2, self.EOS_TAG: 3}
        self.inverse_dict = dict()
        self.count = dict()

    def __len__(self):
        return len(self.dict)

    def fit(self, sentence: list):
        """
        传入句子，统计词频
        :param sentence:
        :return:
        """
        for word in sentence:  # 如果字典里没有某个单词，则其频数为0
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min_count=None, max_count=None, vocab_size=None):
        """
        构造词表
        :param min_count:
        :param max_count:
        :param vocab_size:
        :return:
        """
        # 将字典复制一份，因为在遍历字典时无法对删除字典的内容
        tmp_count = self.count.copy()
        for key, value in tmp_count.items():
            if min_count is not None:  # 满足min_count要求
                if value < min_count:
                    del self.count[key]
                    continue
            if max_count is not None:  # 满足max_count要求
                if value > max_count:
                    del self.count[key]
                    continue
        if vocab_size is not None:  # 满足词表内单词数量要求
            self.count = dict(sorted(self.count.items(), key=lambda x: x[1], reverse=True)[:vocab_size])
        # 构造词典
        for key in self.count:
            self.dict[key] = len(self.dict)
        # 构造反向词典
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence: list, seq_len: int, add_eos=False) -> list:
        """
        将句子转成序列
        训练时，特征值中不需要eos，目标值中需要eos
        如果不需要eos，返回的结果长度为seq_len
        如果需要eos，返回的结果长度为seq_len+1
        结构为 数字字符 + (EOS) + PAD
        :param sentence: 单词组成的列表，即句子
        :param seq_len: 指定序列长度，可能要对句子进行删减或填充
        :param add_eos: 是否添加eos标记
        """
        res_list = []
        if len(sentence) > seq_len:  # 如果句子长度大于seq_len，裁剪
            res_list += sentence[:seq_len]
            if add_eos:
                res_list += [self.EOS_TAG]
        else:
            res_list += sentence
            if add_eos:
                res_list += [self.EOS_TAG]
                res_list += [self.PAD_TAG] * (seq_len + 1 - len(res_list))  # 如果句子长度小于seq_len，填充
            else:
                res_list += [self.PAD_TAG] * (seq_len - len(res_list))
        # 如果词表中不存在某个token，将该token对应的索引换成UNK_TAG对应的索引
        res_list = [self.dict.get(token, self.dict[self.UNK_TAG]) for token in res_list]
        return res_list

    def inverse_transform(self, sequence: list) -> list:
        """
        将序列转成句子
        :param sequence: 序列
        :return: 句子
        """
        ret = list()
        for i in sequence:
            if i == self.dict[self.EOS_TAG]:  # 去除EOS及后面的字符
                break
            ret.append(self.inverse_dict.get(i, self.UNK_TAG))  # 如果不存在，则用UNK代替
        return ret


def save_model(by_char=False, input=False):
    s2s = Sen2Seq()
    # 拼接文件路径、模型路径
    file_path = "./corpus/chatbot/{}_by_{}.txt".format("input" if input else "target", "char" if by_char else "word")
    model_path = "./model/chatbot/s2s_{}_by_{}.pkl".format("input" if input else "target",
                                                           "char" if by_char else "word")
    desc = "Processing {} By {} Model".format("Input" if input else "Target", "Char" if by_char else "Word")
    with open(file_path, mode="r", encoding="utf-8") as f_file:
        for line in tqdm(f_file.readlines(), desc=desc):
            s2s.fit(line.strip().split(" "))  # 必须strip()，因为读到的每行都有换行符
    s2s.build_vocab()
    with open(model_path, mode="wb") as f_model:
        pickle.dump(s2s, f_model)
    print("Vocab_size: {}".format(len(s2s)))


def load_model(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def run():
    for i in range(2):
        for j in range(2):
            save_model(i, j)


if __name__ == '__main__':
    sentence = list("我爱北京天安门，你爱我")
    s2s = Sen2Seq()
    s2s.fit(sentence)
    s2s.build_vocab()
    print(s2s.dict)
    print(s2s.inverse_dict)
    sequence = s2s.transform(sentence, seq_len=15, add_eos=True)
    print(sequence)
    ret = s2s.inverse_transform(sequence)
    print(ret)
