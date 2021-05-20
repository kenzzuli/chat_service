"""
dataset和dataloader
"""
from torch.utils.data import DataLoader, Dataset, random_split
import config
import torch
from config import by_char


class ChatbotDataset(Dataset):
    def __init__(self):
        self.__input_path = config.chatbot_input_by_char_path if by_char else config.chatbot_input_by_word_path
        self.__target_path = config.chatbot_target_by_char_path if by_char else config.chatbot_target_by_word_path
        # 序列最大长度，按字和词不一样
        self.seq_len = config.seq_len_by_char if by_char else config.seq_len_by_word
        # 两个s2s
        self.input_s2s = config.s2s_input_by_char if by_char else config.s2s_input_by_word
        self.target_s2s = config.s2s_target_by_char if by_char else config.s2s_target_by_word
        self.input, self.target = self.get_data()

    def get_data(self):
        with open(self.__input_path, mode="r", encoding="utf8") as f_input:
            input = f_input.readlines()
        with open(self.__target_path, mode="r", encoding="utf8") as f_target:
            target = f_target.readlines()
        assert len(input) == len(target), "Corpus Invalid"
        return input, target

    def __getitem__(self, index):
        input = self.input[index].strip().split()
        target = self.target[index].strip().split()
        input_length = min(len(input), self.seq_len)  # 如果句子长度超过最大句长，会裁剪到最大句长
        target_length = min(len(target), self.seq_len)
        # 将input 和 target转成序列
        input = self.input_s2s.transform(input, seq_len=self.seq_len)
        target = self.target_s2s.transform(target, seq_len=self.seq_len, add_eos=True)
        return input, target, input_length, target_length

    def __len__(self):
        return len(self.input)


def get_dataloader():
    all_dataset = ChatbotDataset()
    train_size = int(0.8 * len(all_dataset))
    test_size = len(all_dataset) - train_size
    train_dataset, test_dataset = random_split(all_dataset, [train_size, test_size])
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.train_batch_size,
                                  collate_fn=collate_fn, shuffle=True, drop_last=config.drop_last)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=config.test_batch_size,
                                 collate_fn=collate_fn, shuffle=True, drop_last=config.drop_last)

    return train_dataloader, test_dataloader


def collate_fn(batch):
    """
        定义整理函数
        :param batch: [(input, target, input_length, target_length), (input, target, input_length, target_length), ...]
        :return:
        """
    # 对batch依据input_length从大到小排序，只有encoder中需要排序
    batch = sorted(batch, key=lambda x: x[2], reverse=True)
    # 先对batch拆包，变成多个元组对象，然后从每个元组中取第一个元素组成元组作为zip的第一个元素，相当于矩阵的转置
    input, target, input_length, target_length = zip(*batch)
    input = torch.LongTensor(input)
    target = torch.LongTensor(target)
    input_length = torch.LongTensor(input_length)
    target_length = torch.LongTensor(target_length)
    return input, target, input_length, target_length
