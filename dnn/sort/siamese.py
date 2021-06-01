"""
构建孪生神经网络
"""
# 1.embedding
# 2.GRU
# 3.attention
# 4.attention concate GRU output
# 5.GRU
# 6.pooling
# 7.DNN
import torch.nn as nn
import torch
import torch.nn.functional as F
import config


class Siamese(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=len(config.sort_ws),
                                      embedding_dim=config.sort_embedding_dim,
                                      padding_idx=config.sort_ws_padding_index)
        self.gru1 = nn.GRU(input_size=config.sort_embedding_dim, hidden_size=config.sort_hidden_size,
                           num_layers=config.sort_num_layers, batch_first=config.sort_batch_first,
                           bidirectional=config.sort_bidirectional, dropout=config.sort_dropout)
        self.gru2 = nn.GRU(input_size=config.sort_hidden_size * config.sort_num_directions * 2,
                           hidden_size=config.sort_hidden_size,
                           num_layers=1,
                           batch_first=config.sort_batch_first,
                           bidirectional=False)
        self.dnn = nn.Sequential(
            nn.Linear(config.sort_num_directions * config.sort_hidden_size * 4, config.sort_linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(config.sort_linear_size),
            nn.Dropout(config.sort_dropout),

            nn.Linear(config.sort_linear_size, config.sort_linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(config.sort_linear_size),
            nn.Dropout(config.sort_dropout),

            nn.Linear(config.sort_linear_size, 2),
        )

    def forward(self, input1, input2):
        """
        :param input1: [batch_size, seq_len]
        :param input2: [batch_size, seq_len]
        :return:
        """
        # 构造两个mask矩阵，位置为pad的地方值为1，位置不为pad的地方值为0
        mask1, mask2 = input1.eq(config.sort_ws_padding_index), input2.eq(config.sort_ws_padding_index)
        input1 = self.embedding(input1)  # [batch_size, seq_len1, embedding_dim]
        input2 = self.embedding(input2)  # [batch_size, seq_len2, embedding_dim]
        # 第一次GRU
        # 这里设置了batch_first = True
        # output1 [batch_size, seq_len1, num_directions*hidden_size]
        # hidden_state [num_layers*num_directions,batch_size, hidden_size]
        output1, hidden_state1 = self.gru1(input1)
        output2, hidden_state2 = self.gru1(input2)
        # 注意力
        output1_align, output2_align = self.soft_attention_align(output1, output2, mask1, mask2)
        # 拼接
        output1 = torch.cat([output1, output1_align], dim=-1)  # [batch_size, seq_len, num_directions*hidden_size*2]
        output2 = torch.cat([output2, output2_align], dim=-1)  # [batch_size, seq_len, num_directions*hidden_size*2]
        # 第二次GRU
        # 这里也设置了batch_first = True
        # output1 [batch_size, seq_len1, num_directions*hidden_size]
        # hidden_state [num_layers*num_directions,batch_size, hidden_size]
        gru2_output1, gru2_hidden_state1 = self.gru2(output1)
        gru2_output2, gru2_hidden_state2 = self.gru2(output2)
        # 池化
        # [batch_size, 2*num_directions*hidden_size]
        output1_pooled = self.apply_pooling(gru2_output1)
        output2_pooled = self.apply_pooling(gru2_output2)
        # [batch_size, 4*num_directions*hidden_size]
        out = torch.cat([output1_pooled, output2_pooled], dim=-1)
        out = self.dnn(out)  # [batch_size, 2]
        return F.softmax(out, dim=-1)

    @staticmethod
    def apply_pooling(output):
        # 将窗口大小设为句长，相当于对整个句子取均值或最大值，达到n_gram的效果, seq_len维度变为1
        # [batch_size, num_directions*hidden_size]
        avg_pooled = F.avg_pool1d(output.transpose(1, 2), kernel_size=output.size(1)).squeeze()
        max_pooled = F.max_pool1d(output.transpose(1, 2), kernel_size=output.size(1)).squeeze()

        # [batch_size, 2*num_directions*hidden_size]
        return torch.cat([avg_pooled, max_pooled], dim=1)

    @staticmethod
    def soft_attention_align(x1, x2, mask1, mask2):
        """
        实现attention
        :param x1 [batch_size, seq_len_1, num_directions*hidden_size]
        :param x2 [batch_size, seq_len_2, num_directions*hidden_size]
        :param mask1 [batch_size, seq_len_1]
        :param mask2 [batch_size, seq_len_2]
        :return output1_align [batch_size, seq_len_2, num_directions*hidden_size]
        :return output2_align [batch_size, seq_len_1, num_directions*hidden_size]
        """
        # 将mask中值为1的地方（即所有的pad）转成-inf，这样在运算时pad就没什么用了
        mask1 = mask1.float().masked_fill_(mask1, float("-inf"))
        mask2 = mask2.float().masked_fill_(mask2, float("-inf"))
        # 1.attention weight
        # 2.attention_weight * output
        # x2.permute(2,1) [batch_size, num_directions*hidden_size, seq_len_2]
        # mask2.unsqueeze(1) [batch_size, 1, seq_len_2]
        # 这里把x2当作encoder x1当作decoder
        # decoder 和 encoder 运算后再softmax得到attention weight
        # attention weight再和encoder运算得到context vector
        x1_attention_energy = x1.bmm(x2.permute(0, 2, 1))  # [batch_size, seq_len_1, seq_len_2]
        x1_attention_weight = F.softmax(x1_attention_energy + mask2.unsqueeze(1), dim=-1)
        output2_align = x1_attention_weight.bmm(x2)  # [batch_size, seq_len_1, num_directions*hidden_size]

        # 把x1当encoder 原理同上
        x2_attention_energy = x1_attention_energy.permute(0, 2, 1)  # [batch_size, seq_len_2, seq_len_1]
        x2_attention_weight = F.softmax(x2_attention_energy + mask1.unsqueeze(1), dim=-1)
        output1_align = x2_attention_weight.bmm(x1)  # [batch_size, seq_len_2, num_directions*hidden_size]
        return output1_align, output2_align
