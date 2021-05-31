"""
测试dnn
"""
from dnn.recall.recall import Recall

if __name__ == '__main__':
    recall = Recall(vectorize_method="fasttext")
    sentence = "蒋夏梦和周瓴？"
    # sentence = "产品经理做什么？"
    print(recall.predict(sentence))
