"""
处理召回需要的数据
{
    "问题1":{
        "主体":["主体1","主体3","主体3"..],
        "问题1分词后的句子":["word1","word2","word3"...],
        "答案":"答案"
    },
    "问题2":{
        ...
    }
}
"""
import config
import pandas as pd
from lib import cut
import json


def _prepare_q_a(q_lines: list, a_lines: list, qa_dict: dict):
    for q, a in zip(q_lines, a_lines):
        q = q.strip()
        a = a.strip()
        qa_dict[q] = dict()
        # 获取命名实体，就是按词分词后，选择词性为kc（课程）的所有词作为命名实体
        ret = cut(q, by_character=False, with_pos=True)
        # [('产品经理', 'kc'), ('的', 'uj'), ('课程', 'n'), ('有', 'v'), ('什么', 'r'), ('特点', 'n'), ('？', 'x')]
        qa_dict[q]["entity"] = [i[0] for i in ret if i[1] == "kc" or i[1] == "shisu"]
        qa_dict[q]["q_cut_by_word"] = [i[0] for i in ret]
        qa_dict[q]["q_cut_by_char"] = cut(q, by_character=True)
        qa_dict[q]["answer"] = a


def prepare_recall_corpus():
    qa_dict = dict()
    # 处理qa txt文档
    with open(config.q_path, mode="r", encoding="utf-8") as q_file:
        with open(config.a_path, mode="r", encoding="utf-8") as a_file:
            q_lines = q_file.readlines()
            a_lines = a_file.readlines()
    assert len(q_lines) == len(a_lines), "question and answer do not match"
    _prepare_q_a(q_lines, a_lines, qa_dict)

    # 处理 excel
    df = pd.read_excel(config.qa_excel_path)
    assert "问题" in df.columns and "答案" in df.columns
    _prepare_q_a(df["问题"], df["答案"], qa_dict)
    # 保存为json
    with open(config.qa_path, mode="w") as qa_file_write:
        json.dump(obj=qa_dict, fp=qa_file_write, sort_keys=True, indent=2, separators=(", ", ": "), ensure_ascii=False)
    # 读取json
    with open(config.qa_path, mode="r") as qa_file_read:
        data = json.load(qa_file_read)
        # 格式化输出，查看结果
        print(json.dumps(data, sort_keys=True, indent=2, separators=(", ", ": "), ensure_ascii=False))
