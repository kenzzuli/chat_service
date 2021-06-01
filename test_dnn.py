"""
测试dnn
"""
from dnn.recall.recall import Recall
from dnn.sort.word_sequence import save_dnn_sort_ws
from dnn.sort.dataset import dnn_dataloader
from dnn.sort.train_and_eval import train, eval
from dnn.sort.sort import DnnSort

if __name__ == '__main__':
    recall = Recall(vectorize_method="fasttext")
    # sentence = "蒋夏梦和周瓴？"
    # sentence = "产品经理做什么？"
    sentence = "上海外国语大学"
    recall_list = recall.predict(sentence)

    # save_dnn_sort_ws()

    # for i in dnn_dataloader:
    #     print(i)
    #     exit()
    # train()
    # eval()
    sort = DnnSort()
    # print(sort.predict("python好学吗？", ["python难吗？", "蒋夏梦是谁？", "c语言好就业吗？"]))
    print(sort.predict(sentence, recall_list))
