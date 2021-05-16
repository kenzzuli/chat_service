import config
from lib import cut
from tqdm import tqdm
import json
import random

# 训练集:测试集 = N:1
N = 4


def keywords_in_line(line):
    """判断一句话中是否有指定的关键词"""
    keywords_list = ["传智播客", "传智", "黑马程序员", "黑马", "python", "人工智能",
                     "c语言", "c++", "java", "javaee", "前端", "移动开发", "ui", "ue",
                     "大数据", "软件测试", "php", "h5", "产品经理", "linux", "运维", "go语言",
                     "区块链", "影视制作", "pmp", "项目管理", "新媒体", "小程序", "前端"]
    for keyword in keywords_list:
        if keyword in line:
            return True
    return False


def process_xiaohuangji(fout_train, fout_test, by_char):
    num = 0
    fin = open(config.xiaohuangji_path, "r").readlines()
    first_m_flag = True  # 标志着是否是第一个m
    for line in tqdm(fin, desc="Processing Xiaohuangji Corpus"):
        if line.startswith("M"):  # 句子以M开头
            if first_m_flag:
                if not keywords_in_line(line):  # 句子不包含指定关键词
                    line = line[2:].strip()  # 去除最开始的M
                    if len(line) > 1:  # 删去句子长度为1的句子
                        line_cut = " ".join(cut(line, by_character=by_char)).strip()  # 将分词后的结果，用空格连接在一起
                        line_cut += "\t__label__chat"  # 添加类别信息
                        # 将数据按照4：1的比例分为训练集和测试集
                        if random.randint(0, N + 1) == 0:
                            fout_test.write(line_cut + "\n")  # 加上换行
                        else:
                            fout_train.write(line_cut + "\n")
                        num += 1
            first_m_flag = not first_m_flag
    return num


def process_by_hand(fout_train, fout_test, by_char):
    """处理手工构造的句子"""
    num = 0
    fin = open(config.by_hand_path, "r").read()  # 读取json文件
    # fin_dic = eval(fin)  # 将文件转成字典
    fin_dic = json.loads(fin)  # 用json直接读也行
    for q_list_list in tqdm(fin_dic.values(), desc="Processing Homemade Corpus"):  # 列表中嵌套列表
        for q_list in q_list_list:
            for q in q_list:
                if "校区" in q:
                    continue
                q = " ".join(cut(q, by_character=by_char)).strip()  # 分词
                q += "\t__label__QA"
                if random.randint(0, N + 1) == 0:
                    fout_test.write(q + "\n")
                else:
                    fout_train.write(q + "\n")
                num += 1
    return num


def process_crawled_corpus(fout_train, fout_test, by_char):
    """处理爬取的数据"""
    num = 0
    fin = open(config.by_crawl_path, "r").readlines()
    for line in tqdm(fin, desc="Processing Crawled Corpus"):
        q = " ".join(cut(line, by_character=by_char)).strip()  # 分词
        q += "\t__label__QA"
        if random.randint(0, N + 1) == 0:
            fout_test.write(q + "\n")
        else:
            fout_train.write(q + "\n")
        num += 1
    return num


def process(by_char=False):
    """
    :param by_char: 语料是否按照字切分
    :return:
    """
    fout_train_path = config.classify_corpus_by_word_train_path if not by_char else config.classify_corpus_by_char_train_path
    fout_test_path = config.classify_corpus_by_word_test_path if not by_char else config.classify_corpus_by_char_test_path
    fout_train = open(fout_train_path, "a")
    fout_test = open(fout_test_path, "a")
    # 处理小黄鸡
    num_chat = process_xiaohuangji(fout_train, fout_test, by_char)
    # 处理手工构造数据
    num_qa = process_by_hand(fout_train, fout_test, by_char)
    # 处理抓取数据
    num_qa += process_crawled_corpus(fout_train, fout_test, by_char)
    print(num_chat, num_qa)
    fout_train.close()
    fout_test.close()


def get_text_and_label(line):
    text, label = [i.strip() for i in line.split("\t")]
    return text, label
