"""
测试dnn
"""
from dnn.recall.recall import Recall
from dnn.sort.word_sequence import save_dnn_sort_ws
from dnn.sort.dataset import dnn_dataloader

if __name__ == '__main__':
    # recall = Recall(vectorize_method="fasttext")
    # sentence = "蒋夏梦和周瓴？"
    # sentence = "产品经理做什么？"
    # print(recall.predict(sentence))

    # save_dnn_sort_ws()
    for i in dnn_dataloader:
        print(i)
        exit()
