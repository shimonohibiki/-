import os
import numpy as np
import matplotlib.pyplot as plt
import sys

def average_5x5_stride5(matrix):
    rows, cols = matrix.shape
    result_rows = (rows + 4) // 5  # 切り上げ除算
    result_cols = (cols + 4) // 5  # 切り上げ除算
    result = np.zeros((result_rows, result_cols))
    
    for i in range(result_rows):
        for j in range(result_cols):
            # 5x5の窓の範囲を定義（ストライド5）
            row_start = i * 5
            row_end = min(row_start + 5, rows)
            col_start = j * 5
            col_end = min(col_start + 5, cols)
            
            # 窓内の値を取得（端の場合は5x5よりも小さくなる可能性がある）
            window = matrix[row_start:row_end, col_start:col_end]
            
            # 窓内の値の平均を計算
            result[i, j] = np.mean(window)
    
    return result


for k in range(13):
    for j in range(17):
        # 入力
        with open(f"C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\experiment-i\\S{k+1}\\{j+1}.txt", "r") as file:
            lines = file.read()
        lines_sp = lines.splitlines()
        
        # 新しいディレクトリを作成
        output_dir = f"C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\heatmap_avg25\\S{k+1}\\{j+1}"
        os.makedirs(output_dir, exist_ok=True)
        
        for i in range(len(lines_sp)):
            # データを64x32の形に整形
            data = [int(x) for x in lines_sp[i].split()]
            matrix = np.array(data).reshape(64, 32)
            
            # 近傍16つのマスを平均
            averaged_matrix = average_5x5_stride5(matrix)
  
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