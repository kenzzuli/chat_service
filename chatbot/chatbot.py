"""chatbot的封装"""
import config
from lib import cut
from config import by_char
import torch
import os
from chatbot.seq2seq import Seq2Seq

model_path = config.chatbot_model_path
optimizer_path = config.chatbot_optimizer_path
s2s_input = config.s2s_input
s2s_target = config.s2s_target
# 使用gpu训练
s2s = Seq2Seq().to(config.device)
optimizer = torch.optim.Adam(s2s.parameters())

# 模型加载
if os.path.exists(model_path) and os.path.exists(optimizer_path):
    s2s.load_state_dict(torch.load(model_path, map_location=config.device))
    optimizer.load_state_dict(torch.load(optimizer_path, map_location=config.device))


class Chatbot:
    def __init__(self):
        # 加载模型
        self.s2s = s2s

    def predict(self, sentence):
        self.s2s.eval()
        # 句子转序列
        sentence = cut(sentence, by_character=by_char)
        feature = s2s_input.transform(sentence, config.seq_len)
        # 构造feature和feature——length
        feature = torch.LongTensor(feature).to(config.device).unsqueeze(0)
        feature_length = torch.LongTensor([min(len(sentence), config.seq_len)]).to(config.device)
        # 预测
        y_predict = self.s2s.evaluate_beam_search(feature, feature_length)
        # 转成句子
        for i in y_predict:
            print("".join(s2s_target.inverse_transform(i)))
