"""
准备数据集
"""
from torch.utils.data import DataLoader, Dataset
import config
import torch


class DNNSortDataset(Dataset):
    def __init__(self):
        self.ws = config.sort_ws
        self.seq_len = config.seq_len
        self.q_lines = open(config.sort_q_path).readlines()
        self.sim_q_lines = open(config.sort_similar_q_path).readlines()
        assert len(self.q_lines) == len(self.sim_q_lines), "DNN Sort中数据长度不一致"

    def __getitem__(self, index):
        q = self.q_lines[index].strip().split()
        sim_q = self.sim_q_lines[index].strip().split()
        q_length = min(len(q), self.seq_len)
        sim_q_length = min(len(sim_q), self.seq_len)
        q = self.ws.transform(q, seq_len=self.seq_len)
        sim_q = self.ws.transform(sim_q, seq_len=self.seq_len)
        return q, sim_q, q_length, sim_q_length

    def __len__(self):
        return len(self.q_lines)


def collate_fn(batch):
    """
    定义整理函数
    :param batch: [(q, sim_q, q_length,  sim_q_length), (q, sim_q, q_length,  sim_q_length), ...]
    :return:
    """
    # 对batch依据input_length从大到小排序，只有encoder中需要排序
    batch = sorted(batch, key=lambda x: x[2], reverse=True)
    # 先对batch拆包，变成多个元组对象，然后从每个元组中取第一个元素组成元组作为zip的第一个元素，相当于矩阵的转置
    q, sim_q, q_length, sim_q_length = zip(*batch)
    q = torch.LongTensor(q)
    sim_q = torch.LongTensor(sim_q)
    q_length = torch.LongTensor(q_length)
    sim_q_length = torch.LongTensor(sim_q_length)
    return q, sim_q, q_length, sim_q_length


dnn_dataloader = DataLoader(dataset=DNNSortDataset(), batch_size=config.sort_batch_size, drop_last=True, shuffle=True,
                            collate_fn=collate_fn)
