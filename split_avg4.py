import os
import numpy as np
import matplotlib.pyplot as plt
import sys



def average_neighbors(matrix):
    rows, cols = matrix.shape
    new_rows, new_cols = rows // 2, cols // 2
    new_matrix = np.zeros((new_rows, new_cols), dtype=np.float32)
    
    for i in range(new_rows):
        for j in range(new_cols):
            new_matrix[i, j] = np.mean(matrix[2*i:2*i+2, 2*j:2*j+2])
    
    return new_matrix

for k in range(13):
    for j in range(17):
        # 入力
        with open(f"C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\experiment-i\\S{k+1}\\{j+1}.txt", "r") as file:
            lines = file.read()
        lines_sp = lines.splitlines()
        
        # 新しいディレクトリを作成
        output_dir = f"C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\heatmap_avg4\\S{k+1}\\{j+1}"
        os.makedirs(output_dir, exist_ok=True)
        
        for i in range(len(lines_sp)):
            # データを64x32の形に整形
            data = [int(x) for x in lines_sp[i].split()]
            matrix = np.array(data).reshape(64, 32)
            
            # 近傍4つのマスを平均
            averaged_matrix = average_neighbors(matrix)
            
            # ヒートマップ画像を作成
            fig = plt.figure(figsize=(8, 6), frameon=False)
            ax = fig.add_axes([0, 0, 1, 1])
            im = ax.imshow(averaged_matrix, cmap='turbo')
            
            # グラフの数値を非表示にする
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 保存
            output_file = os.path.join(output_dir, f"heatmap_{k+1}_{j+1}_{i+1}.png")
            plt.savefig(output_file)
            plt.close()