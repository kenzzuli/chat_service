"""
测试用户词典
"""
import jieba
import config
import logging

# 调整jieba的日志等级
jieba.setLogLevel(logging.INFO)
jieba.load_userdict(config.user_dict_path)


def test_user_dict():
    sentence = "人工智能+python和c++哪个难"
    ret = jieba.lcut(sentence)
    print(ret)
