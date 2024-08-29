# split.pyで作成したヒートマップディレクトリから、クラスタリングができるようにディレクトリ構造を変更
import os
import shutil
from pathlib import Path
import glob

# ソースディレクトリ
src_dir = Path(r"C:\Users\hibik\Desktop\kenkyu\dataset\heatmap")

# クラスごとのインデックス
class_to_idx = {
    "class0": [1, 8, 9, 10, 11, 12, 15, 16, 17],
    "class1": [2, 4, 5, 13],
    "class2": [3, 6, 7]
}

# トレーニングデータセット用ディレクトリ
train_dir = src_dir / "train"
train_dir.mkdir(exist_ok=True, parents=True)

# バリデーションデータセット用ディレクトリ
val_dir = src_dir / "val" 
val_dir.mkdir(exist_ok=True, parents=True)

# テストデータセット用ディレクトリ
test_dir = src_dir / "test" 
test_dir.mkdir(exist_ok=True, parents=True)

# ファイルを分類してコピーする
for i in range(1, 14):
    src_subdir = src_dir / f"S{i}"
    
    if 1 <= i <= 8:  # トレーニングデータセット
    # if i == 1:  # トレーニングデータセット
        target_dir = train_dir
    # elif 10 <= i <= 13:  # バリデーションデータセット
    elif 9 <= i <= 11:  # バリデーションデータセット
        target_dir = val_dir
    elif 12 <= i <= 13:
        target_dir = test_dir

    for class_name, indices in class_to_idx.items():
        for j in indices:
            src_path = src_subdir / str(j)
            dst_path = target_dir / class_name / str(j)
            if src_path.exists():
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

# 画像ファイルをすべて取り出す
for target_dir in [train_dir, val_dir, test_dir]:
    for i in range(3):
        class_dir = target_dir / f"class{i}"
        for j in class_to_idx[f"class{i}"]:
            src_dir = class_dir / str(j)
            if src_dir.exists():
                for file in src_dir.glob("*"):
                    shutil.move(str(file), str(class_dir))
                shutil.rmtree(src_dir)

print("処理が完了しました。")