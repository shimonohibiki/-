# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

import config
from archs import model
from dataset import CUDAPrefetcher, ImageDataset
from utils import load_state_dict, accuracy, Summary, AverageMeter, ProgressMeter
from sklearn.metrics import confusion_matrix
import numpy as np

model_names = sorted(
    name for name in model.__dict__ if name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


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

    # Place all data on the preprocessing data loader
    test_prefetcher = CUDAPrefetcher(test_dataloader, config.device)

    return test_prefetcher


def main() -> None:
    # Initialize the model
    resnet_model = build_model()
    print(f"Build `{config.model_arch_name}` model successfully.")
    resnet_model, _, _, _, _, _ = load_state_dict(resnet_model, config.model_weights_path)
    
    print(f"Load `{config.model_arch_name}` "
          f"model weights `{os.path.abspath(config.model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    resnet_model.eval()

    # Load test dataloader
    test_prefetcher = load_dataset()

    # Calculate how many batches of data are in each Epoch
    batches = len(test_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    acc1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    # acc5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    # progress = ProgressMeter(batches, [batch_time, acc1, acc5], prefix=f"Test: ")
    progress = ProgressMeter(batches, [batch_time, acc1], prefix=f"Test: ")

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    test_prefetcher.reset()
    batch_data = test_prefetcher.next()

    # Get the initialization test time
    end = time.time()
    
    # 予測ラベルと正解ラベルを保存するリスト
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        while batch_data is not None:
            # Transfer in-memory data to CUDA devices to speed up training
            images = batch_data["image"].to(device=config.device, non_blocking=True)
            target = batch_data["target"].to(device=config.device, non_blocking=True)

            # Get batch size
            batch_size = images.size(0)

            # Inference
            output = resnet_model(images)
            
            # 予測と正解ラベルを保存
            _, predicted = torch.max(output, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            # measure accuracy and record loss
            # top1, top5 = accuracy(output, target, topk=(1, 5))
            top1, _ = accuracy(output, target, topk=(1, 5))
            acc1.update(top1[0].item(), batch_size)
            # acc5.update(top5[0].item(), batch_size)

            # Calculate the time it takes to fully train a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Write the data during training to the training log file
            if batch_index % config.test_print_frequency == 0:
                progress.display(batch_index + 1)

            # Preload the next batch of data
            batch_data = test_prefetcher.next()

            # Add 1 to the number of data batches to ensure that the terminal prints data normally
            batch_index += 1
        
    # 混同行列の計算
    cm = confusion_matrix(all_targets, all_predictions)

    # 各クラスの正解率を計算
    class_accuracy = cm.diagonal() / cm.sum(axis=1)

    # 結果の表示
    print("\nClass-wise Accuracy:")
    for i, acc in enumerate(class_accuracy):
        print(f"Class {i}: {acc:.2f}")

    # 全体の正解率
    print(f"\nOverall Accuracy: {acc1.avg:.2f}%")
    print(f"Overall Error: {100 - acc1.avg:.2f}%")

    # print metrics
    print(f"Acc@1 error: {100 - acc1.avg:.2f}%")
    # print(f"Acc@5 error: {100 - acc5.avg:.2f}%")


if __name__ == "__main__":
    main()