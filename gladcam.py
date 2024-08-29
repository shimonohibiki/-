import os
import time
import urllib
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
from torch.utils.data import DataLoader

import config
from archs import model
from dataset import CUDAPrefetcher, ImageDataset
from utils import load_state_dict, accuracy, Summary, AverageMeter, ProgressMeter
from sklearn.metrics import confusion_matrix

# colabをダークモードにしていると、グラフ表示したときに目盛が見えなくなってしまうことに対する対処
sns.set_style("white") 

# ImageNetのラベル情報をダウンロード（必要な場合）
# labels = pickle.load(urllib.request.urlopen('https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl'))

def build_model() -> nn.Module:
    resnet_model = model.__dict__[config.model_arch_name](num_classes=config.model_num_classes)
    resnet_model = resnet_model.to(device=config.device, memory_format=torch.channels_last)
    return resnet_model

def load_dataset() -> CUDAPrefetcher:
    test_dataset = ImageDataset(config.test_image_dir,
                                config.image_size,
                                config.model_mean_parameters,
                                config.model_std_parameters,
                                "Test")
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    test_prefetcher = CUDAPrefetcher(test_dataloader, config.device)
    return test_prefetcher

def main() -> None:
    # Initialize the model
    resnet_model = build_model()
    print(f"Build `{config.model_arch_name}` model successfully.")
    resnet_model, _, _, _, _, _ = load_state_dict(resnet_model, config.model_weights_path)
    
    print(f"Load `{config.model_arch_name}` model weights `{os.path.abspath(config.model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    resnet_model.eval()

    # Load test dataloader
    test_prefetcher = load_dataset()

    # Calculate how many batches of data are in each Epoch
    batches = len(test_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    acc1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(batches, [batch_time, acc1], prefix=f"Test: ")

    # Initialize the data loader and load the first batch of data
    test_prefetcher.reset()
    batch_data = test_prefetcher.next()

    # Get the initialization test time
    end = time.time()
    
    all_predictions = []
    all_targets = []

    batch_index = 0
    with torch.no_grad():
        while batch_data is not None:
            images = batch_data["image"].to(device=config.device, non_blocking=True)
            target = batch_data["target"].to(device=config.device, non_blocking=True)

            batch_size = images.size(0)

            output = resnet_model(images)
            
            _, predicted = torch.max(output, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            top1, _ = accuracy(output, target, topk=(1, 5))
            acc1.update(top1[0].item(), batch_size)

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_index % config.test_print_frequency == 0:
                progress.display(batch_index + 1)

            batch_data = test_prefetcher.next()
            batch_index += 1
        
    cm = confusion_matrix(all_targets, all_predictions)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)

    print("\nClass-wise Accuracy:")
    for i, acc in enumerate(class_accuracy):
        print(f"Class {i}: {acc:.2f}")

    print(f"\nOverall Accuracy: {acc1.avg:.2f}%")
    print(f"Overall Error: {100 - acc1.avg:.2f}%")

    # 混同行列の可視化
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == "__main__":
    main()