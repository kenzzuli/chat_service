"""
分词
"""
import jieba.posseg as psg
import jieba
import logging
import config
import string
from lib import stopwords

# 设置jieba的日志等级
jieba.setLogLevel(logging.INFO)
# 所有的小写字母 abcdefghijklmnopqrstuvwxyz
lower_letters = string.ascii_lowercase + "+"
# 所有的标点符号
punctuations = [" ", "？", "，", "。", "！", "：",
                "?", ",", ".", "!", ":"]
# 加载词典
jieba.load_userdict(config.user_dict_path)


def _cut_sentence_by_word(sentence, with_pos, use_stopwords):
    """
    英汉都按照词来切分
    "python和c++哪个难？" --> ["python","和","c++","哪个", "难", "？"]
    """
    if with_pos:
        ret = psg.lcut(sentence)  # 结果是jieba自定义的pair对象
        ret = [(i.word, i.flag) for i in ret]  # 将pair对象转成元组
        if use_stopwords:
            ret = [i for i in ret if i[0] not in stopwords]
        return ret
    else:
        ret = jieba.lcut(sentence)
        ret = [i.strip() for i in ret]  # 结巴把空格当成一个词
        if use_stopwords:
            ret = [i for i in ret if i not in stopwords]
        return ret


def _cut_sentence_by_character(sentence, use_stopwords):
    """
    汉语按字切分，英语按词切分
    python和c++哪个难？ --> python 和 c++ 哪 个 难 ？
    """
    ret = []
    tmp = ""
    for token in sentence:
        if token.lower() in lower_letters:  # 如果当前字符小写后是小写字母
            tmp += token.lower()  # 放到tmp中存储下
        else:  # 如果当前字符不是小写字母
            if tmp is not "":  # 如果tmp不为"" 说明前一个字符肯定是字母
                ret.append(tmp)  # 将tmp添加到结果中
                tmp = ""  # 将tmp置为""
            if token in punctuations:  # 如果当前字符是标点符号，直接跳过
                continue
            else:
                ret.append(token)  # 如果不是标点符号，添加到结果列表中
    if tmp is not "":  # 遍历结束后，如果tmp有内容，还是要加入列表中
        ret.append(tmp)
    if use_stopwords:
        ret = [i for i in ret if i not in stopwords]
    return ret


def cut(sentence, by_character=False, use_stopwords=False, with_pos=False):
    """
    :param sentence: 句子
    :param by_character: 汉语按照字还是词来切分
    :param use_stopwords: 是否使用停用词
    :param with_pos: 是否返回词性
    :return: 列表
    """
    if by_character:
        assert with_pos is False  # 如果按字切分，就不能再有词性了
        result = _cut_sentence_by_character(sentence, use_stopwords=use_stopwords)
    else:
        result = _cut_sentence_by_word(sentence, with_pos=with_pos, use_stopwords=use_stopwords)
    return result


if __name__ == '__main__':
    sentence = "python和c++哪个难？UI/UE呢？"
    print(_cut_sentence_by_character(sentence))
