# from prepare_corpus.prepare_user_dict.test_user_dict import test_user_dict
# from prepare_corpus.prepare_question_answer_pair.prepare_q_a_pair import process_question_answer_pair
# from lib import cut
# from lib import stopwords
# test_user_dict()
# process_question_answer_pair()
# sentence = "python难不难，很难吗？啊 果真"
# print(cut(sentence, by_character=False, with_pos=True, use_stopwords=True))
# print(stopwords)

# 查看BM25中的中间项效果
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np

k = np.linspace(1, 5, 100)
tf = np.linspace(0, 1, 100)
k, tf = np.meshgrid(k, tf)
z = (k + 1) * tf / (k + tf)
fig = plt.figure(figsize=(10, 10))
axes3d = Axes3D(fig)
axes3d.plot_surface(k, tf, z)
plt.show()
