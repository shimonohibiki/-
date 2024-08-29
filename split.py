# 生データからヒートマップを作製

import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm

for k in range(13):

    for j in range(17):

        # 入力
        with open(f"C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\experiment-i\\S{k+1}\\{j+1}.txt", "r") as file:
            lines = file.read()
        lines_sp = lines.splitlines()

        # print(len(lines_sp))
        # sys.exit()

        # 新しいディレクトリを作成
        output_dir = f"C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\heatmap\\S{k+1}\\{j+1}"
        # ディレクトリが存在しない場合は作成する
        os.makedirs(output_dir, exist_ok=True)

        for i in range(len(lines_sp)):
            
            # データを64x32の形に整形
            data = [int(x) for x in lines_sp[i].split()]
            matrix = np.array(data).reshape(64, 32)

            # ヒートマップ画像を作成
            fig = plt.figure(figsize=(8, 6), frameon=False)
            ax = fig.add_axes([0, 0, 1, 1])
            im = ax.imshow(matrix, cmap='turbo')

            # グラフの数値を非表示にする
            ax.set_xticks([])
            ax.set_yticks([])

            # 保存
            output_file = os.path.join(output_dir, f"heatmap_{k+1}_{j+1}_{i+1}.png")
            plt.savefig(output_file)
            plt.close()
