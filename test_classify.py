"""
测试分类相关的api
"""
from classify.build_model import build_classify_model, get_classify_model, evaluate_model
from classify.classify import Classify

textlist = [
    "你 吃饭 了 吗",
    "java",
    "php",
    "python 好 学 吗",
    "理想 很 骨感 ，现实 很 丰满",
    "今天 天气 非常 好",
    "你 怎么 可以 这样 呢",
    "你 是 谁"
]

if __name__ == '__main__':
    # by_char = True
    # build_classify_model(by_char=by_char)
    # evaluate_model(by_char=by_char)
    classfier = Classify()
    print(classfier.predict("python好学吗"))
    print(classfier.predict("今天天气不错啊"))
