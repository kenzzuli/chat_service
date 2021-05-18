"""
颜文字
"""
import config

with open(config.emoji_path, mode="r", encoding="utf8") as fin:
    emojis = [i.strip() for i in fin.readlines()]
