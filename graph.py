import matplotlib.pyplot as plt
import numpy as np

class0 = [1.00, 1.00, 1.00, 1.00, 0.98]
class1 = [0.90, 0.85, 0.96, 0.94, 0.90]
class2 = [0.83, 0.94, 0.93, 0.98, 0.89]
classes = [class0, class1, class2]
x = [1, 4, 9, 16, 25]

# クラスごとのグラフ
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()  # 2次元配列を1次元に変換

for i, cls in enumerate(classes):
    axes[i].plot(x, cls, "o-")
    axes[i].set_title(f'Class {i}')
    axes[i].set_xlabel('resolution')
    axes[i].set_ylabel('accuracy')
    axes[i].set_ylim([0,1])
    axes[i].set_xticks(x)  # x軸の目盛りを設定
    axes[i].set_xticklabels(x)  # x軸の目盛りのラベルを設定

# 平均値の計算と平均値のグラフ
class_avg = np.mean(classes, axis=0)
axes[3].plot(x, class_avg, "o-")
axes[3].set_title('Average')
axes[3].set_xlabel('resolution')
axes[3].set_ylabel('accuracy')
axes[3].set_ylim([0,1])

plt.tight_layout()
plt.show()