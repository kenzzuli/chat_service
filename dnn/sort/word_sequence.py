"""
准备词典
"""
from chatbot.sentence2sequence import Sen2Seq
import config
from tqdm import tqdm
from lib import cut
import pickle


def save_dnn_sort_ws():
    ws = Sen2Seq()
    f1 = open(config.sort_q_path)
    f2 = open(config.sort_similar_q_path)
    total_lines = f1.readlines() + f2.readlines()
    for line in tqdm(total_lines, ascii=False, desc="Fitting"):
        # 这里读到的每行，早就已经按字切分过了，不用再次分词
        word_list = line.strip().split()
        ws.fit(word_list)
    ws.build_vocab()
    with open(config.sort_ws_path, mode="wb") as f_model:
        pickle.dump(ws, f_model)
    print("Vocab_size: {}".format(len(ws)))
