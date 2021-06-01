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
