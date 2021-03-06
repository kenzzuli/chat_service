"""
把encoder和decoder合并，得到seq2seq模型
"""
import torch.nn as nn
from chatbot.encoder import Encoder
from chatbot.decoder import Decoder


class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input, input_length, target):
        encoder_output, encoder_hidden = self.encoder(input, input_length)
        outputs = self.decoder(encoder_hidden, encoder_output, target)
        return outputs

    def evaluate(self, input, input_length):
        encoder_output, encoder_hidden = self.encoder(input, input_length)
        outputs = self.decoder.evaluation(encoder_hidden, encoder_output)
        return outputs

    def evaluate_beam_search(self, input, input_length):
        encoder_output, encoder_hidden = self.encoder(input, input_length)
        seq = self.decoder.evaluation_beam_search_heapq(encoder_output, encoder_hidden)
        return seq
