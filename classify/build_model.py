import fasttext
import config
from prepare_corpus.prepare_classify_corpus.builid_classify_corpus import get_text_and_label


def build_classify_model(by_char=False):
    """
    模型训练和保存
    :param by_char: 模型是否由按字切分的语料训练
    :return:
    """
    corpus_path = config.classify_corpus_by_word_train_path if not by_char else config.classify_corpus_by_char_train_path
    ft_model = fasttext.train_supervised(corpus_path, wordNgrams=2, epoch=20, minCount=5)
    model_path = config.classify_model_by_word_path if not by_char else config.classify_model_by_char_path
    ft_model.save_model(model_path)


def get_classify_model(by_char=False):
    """加载模型"""
    model_path = config.classify_model_by_word_path if not by_char else config.classify_model_by_char_path
    ft_model = fasttext.load_model(model_path)
    return ft_model


def evaluate_model(by_char=False):
    """模型评估"""
    ft_model = get_classify_model(by_char)
    test_path = config.classify_corpus_by_word_test_path if not by_char else config.classify_corpus_by_char_test_path
    # 读取测试集语料
    with open(test_path, "r") as fin:
        lines = fin.readlines()
    # 统计
    correct = 0
    total = len(lines)
    for line in lines:
        text, label = get_text_and_label(line)
        predict = ft_model.predict(text)[0][0]
        if predict == label:
            correct += 1
    print("Accuracy: {:.6f}%".format(100.0 * correct / total))
