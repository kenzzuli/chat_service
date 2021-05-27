"""chatbot的封装"""
import config
from lib import cut
from config import by_char
import torch
from chatbot.seq2seq import Seq2Seq
import random


class Chatbot:
    def __init__(self):
        # 加载模型
        self.s2s = Seq2Seq().to(config.device)
        self.s2s.load_state_dict(torch.load(config.chatbot_model_path, map_location=config.device))

    def predict(self, sentence):
        self.s2s.eval()
        # 句子转序列
        sentence = cut(sentence, by_character=by_char)
        feature = config.s2s_input.transform(sentence, config.seq_len)
        # 构造feature和feature——length
        feature = torch.LongTensor(feature).to(config.device).unsqueeze(0)
        feature_length = torch.LongTensor([min(len(sentence), config.seq_len)]).to(config.device)
        # 预测
        y_predict = self.s2s.evaluate_beam_search(feature, feature_length)
        # 任选一条转成句子并返回
        return "".join(config.s2s_target.inverse_transform(random.choice(y_predict)))
