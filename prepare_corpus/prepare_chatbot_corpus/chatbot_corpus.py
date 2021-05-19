"""
准备闲聊语料
"""
import config
from tqdm import tqdm
from lib import cut, filters, emojis
import re


def replace_emoji(line: str) -> str:
    """
    去除句子中的颜文字
    :param line: str
    :return: str
    """
    for emoji in emojis:
        if emoji in line:
            line = line.replace(emoji, "")
    return line


def filter_line(line: str) -> bool:
    """
    判断是否过滤掉某一行
    :param line: 要判断的句子
    :return:
    """
    if line in filters:  # 如果该句仅有一个过滤字符
        return True
    elif not re.search(r"[\u4e00-\u9fff]", line):  # 如果该句不包含中文字符
        return True
    elif u"小" in line and u"黄" in line and u"鸡" in line:  # 如果该句包含小黄鸡三个字
        return True
    else:
        return False


def prepare_xiaohuangji(by_char=False):
    """
    准备小黄鸡问答语料
    :param by_char: 是否按照字符切分
    """
    with open(config.xiaohuangji_path, mode="r", encoding="utf-8") as fin:
        with open(config.chatbot_input_by_char_path if by_char else config.chatbot_input_by_word_path,
                  mode="w", encoding="utf-8") as f_input:  # 存储问
            with open(config.chatbot_target_by_char_path if by_char else config.chatbot_target_by_word_path,
                      mode="w", encoding="utf-8") as f_target:  # 存储答
                text = fin.readlines()
                num = 0
                lines = list()  # 临时存储句子
                for line in tqdm(text, desc="Processing Xiaohuangji Corpus"):
                    if line.startswith("E"):
                        continue
                    elif line.startswith("M"):
                        lines.append(replace_emoji(line.strip()[2:]))  # 删去句首的M，并去掉颜文字
                    if len(lines) == 2:
                        # 去除符合过滤规则的句子
                        lines = [" ".join(cut(i, by_character=by_char)) + "\n" for i in lines if not filter_line(i)]
                        # 经过筛选后，如果问答都在，则写入文件
                        if len(lines) == 2:
                            f_input.write(lines[0])
                            f_target.write(lines[1])
                            num += 1
                        # 重新变为空列表
                        lines = list()
                print("{} QA Pairs Write".format(num))
