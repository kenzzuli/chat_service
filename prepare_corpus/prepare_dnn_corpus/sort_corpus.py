import config
import json
from tqdm import tqdm
from lib import cut


def extract_and_cut_question(by_char=True):
    num = 0
    fin = open(config.by_hand_path, "r").read()  # 读取json文件
    fin_dic = json.loads(fin)  # 用json直接读也行
    with open(config.recall_corpus, "w") as fout:
        for q_list_list in tqdm(fin_dic.values(), desc="Processing Homemade Corpus"):  # 列表中嵌套列表
            for q_list in q_list_list:
                for q in q_list:
                    q = " ".join(cut(q, by_character=by_char)).strip()  # 分词
                    fout.write(q + "\n")
                    num += 1
    print(num)
