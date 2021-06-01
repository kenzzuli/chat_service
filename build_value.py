labels = [0, 1]
import random

with open("./corpus/dnn/sort/sim_label.txt", mode="w") as fout:
    for i in range(3500):
        fout.write(str(random.choice(labels)) + "\n")
