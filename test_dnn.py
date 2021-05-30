"""
测试dnn
"""
from dnn.recall.recall import Recall

if __name__ == '__main__':
    recall = Recall()
    sentence = "蒋夏梦和周瓴？"
    print(recall.predict(sentence))
