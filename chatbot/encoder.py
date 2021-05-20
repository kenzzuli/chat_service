"""
编码器
"""
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import config
from config import by_char


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_len = config.seq_len_by_char if by_char else config.seq_len_by_word
        self.embed = nn.Embedding(num_embeddings=len(config.s2s_input_by_char if by_char else config.s2s_input_by_word),
                                  embedding_dim=config.embedding_dim,
                                  padding_idx=config.padding_index)
        self.gru = nn.GRU(input_size=config.embedding_dim, hidden_size=config.encoder_hidden_size,
                          num_layers=config.encoder_num_layers, batch_first=config.encoder_batch_first,
                          dropout=config.encoder_drop_out, bidirectional=config.encoder_bidirectional)

    def forward(self, input, input_length):
        """
        :param input: [batch_size, seq_len]
        :param input_length: batch_size
        :return hidden: [encoder_num_layers*num_directions, batch_size, encoder_hidden_size]
        :return output: [seq_len, batch_size, encoder_hidden_size*num_directions]
        """
        embed = self.embed(input)  # (batch_size, seq_len, embedding_dim)
        embed = embed.permute(1, 0, 2)  # （seq_len, batch_size, embedding_dim)
        # 打包
        embed = pack_padded_sequence(embed, lengths=input_length, batch_first=config.encoder_batch_first)
        output, hidden = self.gru(embed)
        # 解包 其实最后没有用到output，解包也毫无意义，只用了hidden
        output, output_length = pad_packed_sequence(output, batch_first=config.encoder_batch_first,
                                                    padding_value=config.padding_index,
                                                    total_length=self.seq_len)
        return output, hidden
