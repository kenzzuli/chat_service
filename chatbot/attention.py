"""
实现attention
"""
import torch
import torch.nn as nn
import config
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, method="general"):
        super(Attention, self).__init__()
        assert method in ["general", "dot", "concat"], "Method Error"
        self.method = method
        if self.method == "dot":
            self.fc_dot = nn.Linear(config.decoder_hidden_size,
                                    config.encoder_hidden_size * config.encoder_num_directions)
        elif self.method == "general":
            self.Wa = nn.Linear(config.encoder_hidden_size * config.encoder_num_directions, config.decoder_hidden_size)
        elif self.method == "concat":
            self.fc_concat = nn.Linear(config.decoder_hidden_size,
                                       config.encoder_hidden_size * config.encoder_num_directions)

    def forward(self, decoder_hidden_state, encoder_outputs):
        """
        :param decoder_hidden_state: [num_layers*decoder_num_directions, batch_size, decoder_hidden_size]
        :param encoder_outputs:[seq_len, batch_size, encoder_hidden_size*encoder_num_directions]
        :return: attention_weight:[batch_size, seq_len]
        """
        # 1.dot
        if self.method == "dot":
            # [num_layers*num_directions, batch_size, decoder_hidden_size] --> [1, batch_size, decoder_hidden_size]
            decoder_hidden_state = decoder_hidden_state[-1, :, :]
            # [1,batch_size, decoder_hidden_size] --> [batch_size, decoder_hidden_size, 1]
            decoder_hidden_state = decoder_hidden_state.permute(1, 2, 0)

            # 为了兼容decoder_hidden_size和encoder_hidden_size不一样的情况
            # [batch_size, decoder_hidden_size, 1] --> [batch_size, decoder_hidden_size]
            decoder_hidden_state = decoder_hidden_state.squeeze(-1)
            # [batch_size, decoder_hidden_size] --> [batch_size, encoder_hidden_size * encoder_num_directions]
            decoder_hidden_state = self.fc_dot(decoder_hidden_state)

            # [seq_len, batch_size, encoder_hidden_size*encoder_num_directions] -->
            # [batch_size, seq_len, encoder_hidden_size*encoder_num_directions]
            encoder_outputs = encoder_outputs.permute(1, 0, 2)
            # 三维矩阵乘法 形状变为[batch_size, seq_len, 1]
            ret = encoder_outputs.bmm(decoder_hidden_state)
            # 去掉最后一个维度 [batch_size, seq_len, 1] --> [batch_size, seq_len]
            attention_energy = ret.squeeze(-1)
            attention_weight = F.softmax(attention_energy, dim=-1)  # 形状不变，仍是[batch_size, seq_len]
        elif self.method == "general":
            # [seq_len, batch_size, encoder_hidden_size*encoder_num_directions] -->
            # [batch_size, seq_len, encoder_hidden_size*encoder_num_directions]
            encoder_outputs = encoder_outputs.permute(1, 0, 2)
            # [batch_size, seq_len, encoder_hidden_size*encoder_num_directions] -->
            # [batch_size, seq_len, decoder_hidden_size]
            encoder_outputs = self.Wa(encoder_outputs)

            # [num_layers*num_directions, batch_size, decoder_hidden_size] --> [1, batch_size, decoder_hidden_size]
            decoder_hidden_state = decoder_hidden_state[-1, :, :]
            # [1,batch_size, decoder_hidden_size] --> [batch_size, decoder_hidden_size, 1]
            decoder_hidden_state = decoder_hidden_state.permute(1, 2, 0)

            # 变为[batch_size, seq_len, 1]
            ret = encoder_outputs.bmm(decoder_hidden_state)
            # 去掉最后一个维度 [batch_size, seq_len, 1] --> [batch_size, seq_len]
            attention_energy = ret.squeeze(-1)
            attention_weight = F.softmax(attention_energy, dim=-1)  # 形状不变，仍是[batch_size, seq_len]
        elif self.method == "concat":
            # [num_layers*num_directions, batch_size, decoder_hidden_size] --> [1, batch_size, decoder_hidden_size]
            decoder_hidden_state = decoder_hidden_state[-1, :, :]
            # [1, batch_size, decoder_hidden_size] --> [batch_size, decoder_hidden_size]
            decoder_hidden_state = decoder_hidden_state.squeeze(0)

            # [batch_size, decoder_hidden_size] --> [batch_size, seq_len, decoder_hidden_size]
            # decoder_hidden_state = decoder_hidden_state.repeat(1, encoder_outputs.size(0), 1)
            decoder_hidden_state = decoder_hidden_state.repeat(encoder_outputs.size(0), 1, 1).permute(1, 0, 2)

            # [seq_len, batch_size, encoder_hidden_size*encoder_num_directions] -->
            # [batch_size, seq_len, encoder_hidden_size*encoder_num_directions]
            encoder_outputs = encoder_outputs.permute(1, 0, 2)
            # 形状变为[batch_size, seq_len, encoder_hidden_size*encoder_num_directions+decoder_hidden_size]
            concated = torch.cat([decoder_hidden_state, encoder_outputs], dim=-1)

