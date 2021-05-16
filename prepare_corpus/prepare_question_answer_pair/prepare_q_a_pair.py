"""
处理问答对
"""
import pandas as pd
import config


def process_question_answer_pair():
    file = pd.read_excel(config.q_a_pair_path)
    assert "问题" and "答案" in file.columns
    for q, a in zip(file["问题"], file["答案"]):
        print(q, a)
