"""
dataset和dataloader
"""
from torch.utils.data import DataLoader, Dataset
import config


class ChatbotDataset(Dataset):
    def __init__(self, by_char=True):
        self.__input_path = config.chatbot_input_by_char_path if by_char else config.chatbot_input_by_word_path
        self.__target_path = config.chatbot_target_by_char_path if by_char else config.chatbot_target_by_word_path
        self.input_s2s = config.s2s_input_by_char if by_char else config.s2s_input_by_word
        self.target_s2s = config.s2s_target_by_char if by_char else config.s2s_target_by_word
        self.input, self.target = self.get_data()

    def get_data(self):
        with open(self.__input_path, mode="r", encoding="utf8") as f_input:
            input = f_input.readlines()
        with open(self.__target_path, mode="r", encoding="utf8") as f_target:
            target = f_target.readlines()
        return input, target

    def __getitem__(self, index):
        input = self.input[index].strip().split()
        target = self.target[index].strip().split()
        input_length = len(input)
        target_length = len(target)
        # 将input 和 target转成序列
        input = self.input_s2s.transform(input, seq_len=config.seq_len)
        target = self.target_s2s.transform(target, seq_len=config.seq_len, add_eos=True)
        return input, target, input_length, target_length

    def __len__(self):
        return len(self.input)
