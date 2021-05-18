"""
准备闲聊语料
"""
import config
from tqdm import tqdm
from lib import cut, filters
import re


def filter_line(line):
    """
    是否过滤掉某一行
    :param line:
    :return:
    """
    if line in filters:  # 如果该句仅有一个过滤字符
        return True
    elif not re.search(r"[\u4e00-\u9fff]", line):  # 如果该句不包含中文字符
        return True
    else:
        return False


def prepare_xiaohuangji(by_char=False):
    with open(config.xiaohuangji_path, mode="r", encoding="utf-8") as fin:
        with open(config.chatbot_input_by_char_path if by_char else config.chatbot_input_by_word_path, mode="w+b",
                  ) as f_input:  # 存储问
            with open(config.chatbot_target_by_char_path if by_char else config.chatbot_target_by_word_path, mode="w+b",
                      ) as f_target:  # 存储答
                text = fin.readlines()
                first_m_flag = True  # 标志是否为 问
                skip_next = False  # 标志是否跳过下一句
                line_length = None  # 标志 问 的长度
                for line in tqdm(text, desc="Processing Xiaohuangji Corpus"):
                    # 是否跳过下一行
                    if skip_next:
                        skip_next = False
                        continue
                    line = line.strip()
                    # 数据清洗
                    # 去除以E开始的行
                    if line.startswith("E"):
                        continue
                    elif line.startswith("M"):
                        line = line[2:]  # 去除M和空格
                        # 按照过滤规则过滤
                        if filter_line(line):
                            if first_m_flag:  # 如果是 问 不满足条件，直接跳过 问 和 答
                                skip_next = True  # 标记跳过下一行
                                continue  # 跳过本行
                            else:  # 如果是 答 不满足条件
                                f_input.seek(-1 * line_length, 1)  # 除了要跳过 答，还有将已经写入的那句 问 给覆盖掉
                                first_m_flag = True  # 重新置为True
                                continue  # 跳过本行
                        line = (" ".join(cut(line, by_character=by_char)) + "\n").encode()  # 分词
                        if first_m_flag:  # 问
                            f_input.write(line)
                            line_length = len(line)  # 记录line的长度
                            first_m_flag = False
                        else:  # 答
                            f_target.write(line)
                            first_m_flag = True
