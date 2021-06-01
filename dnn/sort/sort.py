"""
排序代码封装
"""
from dnn.sort.siamese import Siamese
from lib import cut
import config
import json
import torch


class DnnSort(object):
    def __init__(self):
        self.model = Siamese().to(config.device)
        self.model.load_state_dict(torch.load(config.dnn_model_path, map_location=config.device))
        self.model.eval()
        self.ws = config.sort_ws
        self.qa_dict = json.load(open(config.qa_path, "r"))

    def predict(self, sentence, recall_list):
        # 将用户提问的问题广播到和recall_list相同的数量
        sentence_list = [sentence] * len(recall_list)
        sentence_cut = [cut(i, by_character=True) for i in sentence_list]
        recall_cut = [cut(i, by_character=True) for i in recall_list]
        # [['python', '好', '学', '吗'], ['python', '好', '学', '吗'], ['python', '好', '学', '吗']]
        # [['python', '难', '吗'], ['蒋', '夏', '梦', '是', '谁'], ['c', '语', '言', '好', '就', '业', '吗']]
        sentence_cut = [self.ws.transform(i, config.seq_len) for i in sentence_cut]
        recall_cut = [self.ws.transform(i, config.seq_len) for i in recall_cut]
        q1 = torch.LongTensor(sentence_cut)
        q2 = torch.LongTensor(recall_cut)
        out = self.model(q1, q2)  # [batch_size, 2] 最后一列是句子匹配的概率
        value, index = torch.topk(out[:, -1], k=1, dim=0)
        value = value.item()
        index = index.item()
        # 设置阈值
        if value > config.sort_threshold:  # 如果符合阈值要求，则返回该问题对应的答案
            return self.qa_dict[recall_list[index]]["answer"]
        else:
            return "这个问题我也还没学到啊!"
