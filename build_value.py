"""
在训练孪生神经网络时，只有q和sim_q，没有similarity，所以随机造了一点
"""
labels = [0, 1]
import random

with open("./corpus/dnn/sort/sim_label.txt", mode="w") as fout:
    for i in range(3500):
        fout.write(str(random.choice(labels)) + "\n")
