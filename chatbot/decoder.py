"""
解码器
"""
import torch.nn as nn
import torch
import config
import torch.nn.functional as F
import random
from chatbot.attention import Attention
import heapq


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_len = config.seq_len
        self.embedding = nn.Embedding(
            num_embeddings=len(config.s2s_target),
            embedding_dim=config.embedding_dim,
            padding_idx=config.padding_index)
        self.gru = nn.GRU(input_size=config.embedding_dim, hidden_size=config.decoder_hidden_size,
                          num_layers=config.decoder_num_layers, batch_first=config.decoder_batch_first,
                          dropout=config.decoder_drop_out, bidirectional=config.decoder_bidirectional)
        # 经过全连接层，将[batch_size, hidden_size*num_directions] 转成 [batch_size, vocab_size]
        self.fc = nn.Linear(in_features=config.decoder_hidden_size * config.decoder_num_directions,
                            out_features=len(config.s2s_target))
        self.attention = Attention(method="general")
        self.Wa = nn.Linear(
            config.encoder_hidden_size * config.encoder_num_directions + config.decoder_hidden_size * config.decoder_num_directions,
            config.decoder_hidden_size, bias=False)

    def forward(self, encoder_hidden, encoder_outputs, target):
        """
        :param encoder_hidden: [num_layers*num_directions, batch_size, hidden_size]
        :param target: [batch_size, seq_len+1] 构造数据集时指定长度为seq_len+1, 做teacher forcing
        :return: outputs: [seq_len, batch_size, vocab_size]
        """

        # 1. 接收encoder的hidden_state作为decoder第一个时间步的hidden_state
        decoder_hidden = encoder_hidden
        # 2. 构造第一个时间步的输入 形状为[batch_size, 1]，全为SOS
        batch_size = encoder_hidden.size(1)
        # decoder_input = torch.LongTensor(torch.ones([batch_size, 1]) * config.sos_index)
        decoder_input = torch.LongTensor([[config.sos_index]] * batch_size).to(config.device)

        # 保存所有的结果，需要用outputs和target计算损失
        outputs = []  # outputs最后的形状是 [seq_len, batch_size, vocab_size]
        # 在训练中，是否使用teacher forcing教师纠偏
        use_teacher_forcing = random.random() < config.teacher_forcing_ratio
        if use_teacher_forcing:  # 如果使用教师纠偏，整个batch的输入使用真实值
            for i in range(self.seq_len + 1):
                # 3. 获取第一个时间步的decoder_output，形状为[batch_size, vocab_size]
                decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
                outputs.append(decoder_output)
                # 4. 计算第一个decoder_output，得到最后的输出结果 形状为[batch_size, 1]
                decoder_input = target[:, i].unsqueeze(1)  # 增加一个维度，由[batch_size,]变成[batch_size,1]
        else:  # 如果不使用教师纠偏，整个batch使用预测值
            for i in range(self.seq_len + 1):
                # 3. 获取第一个时间步的decoder_output，形状为[batch_size, vocab_size]
                decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
                outputs.append(decoder_output)
                # 4. 计算第一个decoder_output，得到最后的输出结果 形状为[batch_size, 1]
                # 获取最后一个维度中最大值所在的位置，即确定是哪一个字符，以此作为下一个时间步的输入
                # 模型最开始也不知道哪个位置对应哪个字符，通过训练，慢慢调整，才知道的
                decoder_input = torch.argmax(decoder_output, dim=-1, keepdim=True)  # [batch_size, 1]
                # 5. 把前一次的hidden_state作为当前时间步的hidden_state，把前一次的输出作为当前时间步的输入
                # 6. 循环4-5
        # 将outputs列表在第0个维度累加
        outputs = torch.stack(outputs, dim=0)  # [seq_len, batch_size, vocab_size]
        return outputs

    def forward_step(self, decoder_input, decoder_hidden, encoder_outputs):
        """
        计算每个时间步的结果
        :param decoder_input [batch_size,1]
        :param decoder_hidden [num_layers*num_directions, batch_size, hidden_size]
        :param encoder_outputs [seq_len, batch_size, encoder_hidden_size*encoder_num_directions]
        :return output [batch_size, vocab_size]
        :return decoder_hidden 形状同上面的decoder_hidden
        """
        decoder_input_embed = self.embedding(decoder_input)  # [batch_size, 1, embedding_dim]
        decoder_input_embed = decoder_input_embed.permute(1, 0, 2)  # [1, batch_size, embedding_dim]
        output, decoder_hidden = self.gru(decoder_input_embed, decoder_hidden)
        # output: [1, batch_size, decoder_hidden_size*decoder_num_directions]
        # decoder_hidden: [decoder_num_layers*decoder_num_directions, batch_size, hidden_size]

        #### 增加attention #####
        # 形状为[batch_size, seq_len]
        attention_weight = self.attention(decoder_hidden, encoder_outputs)
        # 形状为[batch_size, 1, seq_len]
        attention_weight = attention_weight.unsqueeze(1)
        # 形状为[batch_size, seq_len, encoder_hidden_size*encoder_num_directions]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # 形状为[batch_size, 1, encoder_hidden_size*encoder_num_directions]
        context_vector = attention_weight.bmm(encoder_outputs)
        # 形状为[batch_size, 1, encoder_hidden_size*encoder_num_directions+decoder_hidden*decoder_num_directions]
        concated = torch.cat([output.permute(1, 0, 2), context_vector], dim=-1)
        # 形状为[batch_size, encoder_hidden_size*encoder_num_directions+decoder_hidden*decoder_num_directions]
        concated = concated.squeeze(1)
        # 形状为[batch_size, decoder_hidden_size]
        output = torch.tanh(self.Wa(concated))
        ######attention结束#######

        # 将output的第0个维度去掉
        # output = output.squeeze(0)  # [batch_size, hidden_size*num_directions]
        output = self.fc(output)  # [batch_size, vocab_size]
        output = F.log_softmax(output, dim=-1)  # 取概率 [batch_size, vocab_size]
        return output, decoder_hidden

    def evaluation(self, encoder_hidden, encoder_outputs):
        """
        模型评估时调用
        :param encoder_hidden: [num_direction*num_layers, batch_size, hidden_size]
        :return outputs: [seq_len, batch_size, vocab_size]
        """
        decoder_hidden = encoder_hidden  # 获取encoder_hidden作为初始的decoder_hidden
        batch_size = encoder_hidden.size(1)  # 获取batch_size,构造初始的decoder_input
        decoder_input = torch.LongTensor([[config.sos_index]] * batch_size).to(config.device)
        outputs = []

        while True:
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            outputs.append(decoder_output)
            decoder_input = torch.argmax(decoder_output, dim=-1, keepdim=True)
            # 构造一个形状为[batch_size, 1]全为eos或pad的tensor
            eos_tensor = torch.LongTensor([[config.eos_index]] * batch_size).to(config.device)
            pad_tensor = torch.LongTensor([[config.padding_index]] * batch_size).to(config.device)
            # 如果整个batch全部预测的都是eos或pad，则结束循环
            if eos_tensor.eq(decoder_input).all().item() or pad_tensor.eq(decoder_input).all().item():
                break

        outputs = torch.stack(outputs, dim=0)
        return outputs

    # decoder中的新方法
    def evaluation_beam_search_heapq(self, encoder_outputs, encoder_hidden):
        """使用 堆 来完成beam search，对是一种优先级的队列，按照优先级顺序存取数据"""

        batch_size = encoder_hidden.size(1)
        # 1. 构造第一次需要的输入数据，保存在堆中
        decoder_input = torch.LongTensor([[config.sos_index] * batch_size]).to(config.device)
        decoder_hidden = encoder_hidden  # 需要输入的hidden

        prev_beam = Beam()
        prev_beam.add(1, False, [decoder_input.item()], decoder_input, decoder_hidden)
        while True:
            cur_beam = Beam()
            # 2. 取出堆中的数据，进行forward_step的操作，获得当前时间步的output，hidden
            # 这里使用下划线进行区分
            all_complete_num = 0
            for _probability, _complete, _seq, _decoder_input, _decoder_hidden in prev_beam:
                # 判断前一次的_complete是否为True，如果是，则不需要forward
                # 有可能为True，但是概率并不是最大
                if _complete is True:
                    cur_beam.add(_probability, _complete, _seq, _decoder_input, _decoder_hidden)
                    all_complete_num += 1
                else:
                    decoder_output_t, decoder_hidden = self.forward_step(_decoder_input, _decoder_hidden,
                                                                         encoder_outputs)
                    value, index = torch.topk(decoder_output_t, config.beam_width)  # [batch_size=1,beam_width=3]
                    # 3. 从output中选择topk（k=beam width）个输出，作为下一次的input
                    for m, n in zip(value[0], index[0]):
                        decoder_input = torch.LongTensor([[n]]).to(config.device)
                        seq = _seq + [n.item()]
                        probability = _probability * m
                        if n.item() == config.eos_index:
                            complete = True
                        else:
                            complete = False
                        # 4. 把下一个时间步中需要的输入等数据保存在一个新的堆中
                        cur_beam.add(probability, complete, seq, decoder_input, decoder_hidden)

            # 5. 获取新的堆中的优先级最高（概率最大）的数据，判断数据是否是EOS结尾或者是否达到最大长度，如果是，停止迭代
            best_prob, best_complete, best_seq, _, _ = max(cur_beam)
            seq_len = config.seq_len
            # if best_complete is True or len(best_seq) - 1 == seq_len:  # 减去sos
            if len(best_seq) - 1 == seq_len or all_complete_num == config.beam_width:
                ret = []
                for _, _, sequence, _, _ in cur_beam:
                    ret.append(self._prepare_seq(sequence))
                return ret

            else:
                # 6. 则重新遍历新的堆中的数据
                prev_beam = cur_beam

    @staticmethod
    def _prepare_seq(seq):  # 对结果进行基础的处理，共后续转化为文字使用
        if seq[0] == config.sos_index:
            seq = seq[1:]
        if seq[-1] == config.eos_index:
            seq = seq[:-1]
        return seq


class Beam:
    def __init__(self):
        self.heap = list()  # 保存数据的位置
        self.beam_width = config.beam_width  # 保存数据的总数

    def add(self, probability, complete, seq, decoder_input, decoder_hidden):
        """
        添加数据，同时判断总的数据个数，多则删除
        :param probability: 概率乘积
        :param complete: 最后一个是否为EOS
        :param seq: list，所有token的列表
        :param decoder_input: 下一次进行解码的输入，通过前一次获得
        :param decoder_hidden: 下一次进行解码的hidden，通过前一次获得
        :return:
        """
        heapq.heappush(self.heap, [probability, complete, seq, decoder_input, decoder_hidden])
        # 判断数据的个数，如果大，则弹出。保证数据总个数小于等于3
        if len(self.heap) > self.beam_width:
            heapq.heappop(self.heap)

    def __iter__(self):  # 让该beam能够被迭代
        return iter(self.heap)
