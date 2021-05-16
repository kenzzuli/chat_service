"""
获取停用词
"""
import config

with open(config.stopwords_path, mode="r", encoding="utf8") as fin:
    stopwords = [i.strip() for i in fin.readlines()]

