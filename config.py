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
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# Model arch name
model_arch_name = "resnet18"
# Model normalization parameters
model_mean_parameters = [0.485, 0.456, 0.406] #これは各色チャンネル（RGB）の平均値を表します。
model_std_parameters = [0.229, 0.224, 0.225] #これは各色チャンネル（RGB）の標準偏差を表します。
# Model number class
model_num_classes = 3

# Current configuration parameter method
# mode = "train"
mode = "test"

# 生データか、低解像度データか

## rawdata　これは生データ
resolution = "raw"
exp_name = f"{model_arch_name}-sleep_pose"

## resolution4 これは近傍４マスの平均を取っているやつ
# resolution = "avg4"
# exp_name = f"{model_arch_name}-sleep_pose_avg4"

# resolution = "avg9"
# exp_name = f"{model_arch_name}-sleep_pose_avg9"

# resolution = "avg16"
# exp_name = f"{model_arch_name}-sleep_pose_avg16"

# resolution = "avg25"
# exp_name = f"{model_arch_name}-sleep_pose_avg25"

# resolution = "avg36"
# exp_name = f"{model_arch_name}-sleep_pose_avg36"

if resolution == "raw":
    if mode == "train":
        # Dataset address
        train_image_dir = "C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\heatmap\\train"
        valid_image_dir = "C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\heatmap\\val"

        image_size = 224
        batch_size = 128
        num_workers = 4

        # The address to load the pretrained model
        # pretrained_model_weights_path = "./results/pretrained_models/ResNet18-ImageNet_1K-57bb63e.pth.tar"
        pretrained_model_weights_path = ""

        # Incremental training and migration training
        resume = ""

        # Total num epochs
        epochs = 200

        # Loss parameters
        loss_label_smoothing = 0.1
        loss_weights = 1.0

        # Optimizer parameter
        model_lr = 0.01
        model_momentum = 0.9
        model_weight_decay = 2e-05
        model_ema_decay = 0.99998

        # Learning rate scheduler parameter
        lr_scheduler_T_0 = epochs // 4
        lr_scheduler_T_mult = 2
        lr_scheduler_eta_min = 1e-6

        # How many iterations to print the training/validate result
        train_print_frequency = 200
        valid_print_frequency = 20

    if mode == "test":
        # Test data address
        test_image_dir = "C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\heatmap\\test"

        # Test dataloader parameters
        image_size = 224
        batch_size = 256 #全部で4,000のデータセットがあったとすれば、400ずつに分けたときの400がバッチサイズとなります。
        num_workers = 4

        # How many iterations to print the testing result
        test_print_frequency = 1
        model_weights_path = "C:/Users/hibik/Desktop/kenkyu/results/resnet18-sleep_pose/best.pth.tar"

if resolution == "avg4":
    if mode == "train":
        # Dataset address
        train_image_dir = "C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\heatmap_avg4\\train"
        valid_image_dir = "C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\heatmap_avg4\\val"

        image_size = 224
        batch_size = 128
        num_workers = 4

        # The address to load the pretrained model
        # pretrained_model_weights_path = "./results/pretrained_models/ResNet18-ImageNet_1K-57bb63e.pth.tar"
        pretrained_model_weights_path = ""

        # Incremental training and migration training
        resume = ""

        # Total num epochs
        epochs = 200

        # Loss parameters
        loss_label_smoothing = 0.1
        loss_weights = 1.0

        # Optimizer parameter
        model_lr = 0.01
        model_momentum = 0.9
        model_weight_decay = 2e-05
        model_ema_decay = 0.99998

        # Learning rate scheduler parameter
        lr_scheduler_T_0 = epochs // 4
        lr_scheduler_T_mult = 2
        lr_scheduler_eta_min = 1e-6

        # How many iterations to print the training/validate result
        train_print_frequency = 200
        valid_print_frequency = 20

    if mode == "test":
        # Test data address
        test_image_dir = "C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\heatmap_avg4\\test"

        # Test dataloader parameters
        image_size = 224
        batch_size = 256 #全部で4,000のデータセットがあったとすれば、400ずつに分けたときの400がバッチサイズとなります。
        num_workers = 4

        # How many iterations to print the testing result
        test_print_frequency = 1
        model_weights_path = "C:/Users/hibik/Desktop/kenkyu/results/resnet18-sleep_pose_avg4/best.pth.tar"
        
if resolution == "avg9":
    if mode == "train":
        # Dataset address
        train_image_dir = "C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\heatmap_avg9\\train"
        valid_image_dir = "C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\heatmap_avg9\\val"

        image_size = 224
        batch_size = 128
        num_workers = 4

        # The address to load the pretrained model
        # pretrained_model_weights_path = "./results/pretrained_models/ResNet18-ImageNet_1K-57bb63e.pth.tar"
        pretrained_model_weights_path = ""

        # Incremental training and migration training
        resume = ""

        # Total num epochs
        epochs = 200

        # Loss parameters
        loss_label_smoothing = 0.1
        loss_weights = 1.0

        # Optimizer parameter
        model_lr = 0.01
        model_momentum = 0.9
        model_weight_decay = 2e-05
        model_ema_decay = 0.99998

        # Learning rate scheduler parameter
        lr_scheduler_T_0 = epochs // 4
        lr_scheduler_T_mult = 2
        lr_scheduler_eta_min = 1e-6

        # How many iterations to print the training/validate result
        train_print_frequency = 200
        valid_print_frequency = 20

    if mode == "test":
        # Test data address
        test_image_dir = "C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\heatmap_avg9\\test"

        # Test dataloader parameters
        image_size = 224
        batch_size = 256 #全部で4,000のデータセットがあったとすれば、400ずつに分けたときの400がバッチサイズとなります。
        num_workers = 4

        # How many iterations to print the testing result
        test_print_frequency = 1
        model_weights_path = "C:/Users/hibik/Desktop/kenkyu/results/resnet18-sleep_pose_avg9/best.pth.tar"

if resolution == "avg16": 
    if mode == "train":
        # Dataset address
        train_image_dir = "C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\heatmap_avg16\\train"
        valid_image_dir = "C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\heatmap_avg16\\val"

        image_size = 224
        batch_size = 128
        num_workers = 4

        # The address to load the pretrained model
        # pretrained_model_weights_path = "./results/pretrained_models/ResNet18-ImageNet_1K-57bb63e.pth.tar"
        pretrained_model_weights_path = ""

        # Incremental training and migration training
        resume = ""

        # Total num epochs
        epochs = 200

        # Loss parameters
        loss_label_smoothing = 0.1
        loss_weights = 1.0

        # Optimizer parameter
        model_lr = 0.01
        model_momentum = 0.9
        model_weight_decay = 2e-05
        model_ema_decay = 0.99998

        # Learning rate scheduler parameter
        lr_scheduler_T_0 = epochs // 4
        lr_scheduler_T_mult = 2
        lr_scheduler_eta_min = 1e-6

        # How many iterations to print the training/validate result
        train_print_frequency = 200
        valid_print_frequency = 20

    if mode == "test":
        # Test data address
        test_image_dir = "C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\heatmap_avg16\\test"

        # Test dataloader parameters
        image_size = 224
        batch_size = 256 #全部で4,000のデータセットがあったとすれば、400ずつに分けたときの400がバッチサイズとなります。
        num_workers = 4

        # How many iterations to print the testing result
        test_print_frequency = 1
        model_weights_path = "C:/Users/hibik/Desktop/kenkyu/results/resnet18-sleep_pose_avg16/best.pth.tar"

if resolution == "avg25":  
    if mode == "train":
        # Dataset address
        train_image_dir = "C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\heatmap_avg25\\train"
        valid_image_dir = "C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\heatmap_avg25\\val"

        image_size = 224
        batch_size = 128
        num_workers = 4

        # The address to load the pretrained model
        # pretrained_model_weights_path = "./results/pretrained_models/ResNet18-ImageNet_1K-57bb63e.pth.tar"
        pretrained_model_weights_path = ""

        # Incremental training and migration training
        resume = ""

        # Total num epochs
        epochs = 200

        # Loss parameters
        loss_label_smoothing = 0.1
        loss_weights = 1.0

        # Optimizer parameter
        model_lr = 0.01
        model_momentum = 0.9
        model_weight_decay = 2e-05
        model_ema_decay = 0.99998

        # Learning rate scheduler parameter
        lr_scheduler_T_0 = epochs // 4
        lr_scheduler_T_mult = 2
        lr_scheduler_eta_min = 1e-6

        # How many iterations to print the training/validate result
        train_print_frequency = 200
        valid_print_frequency = 20

    if mode == "test":
        # Test data address
        test_image_dir = "C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\heatmap_avg25\\test"

        # Test dataloader parameters
        image_size = 224
        batch_size = 256 #全部で4,000のデータセットがあったとすれば、400ずつに分けたときの400がバッチサイズとなります。
        num_workers = 4

        # How many iterations to print the testing result
        test_print_frequency = 1
        model_weights_path = "C:/Users/hibik/Desktop/kenkyu/results/resnet18-sleep_pose_avg25/best.pth.tar"

if resolution == "avg36":  
    if mode == "train":
        # Dataset address
        train_image_dir = "C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\heatmap_avg36\\train"
        valid_image_dir = "C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\heatmap_avg36\\val"

        image_size = 224
        batch_size = 128
        num_workers = 4

        # The address to load the pretrained model
        # pretrained_model_weights_path = "./results/pretrained_models/ResNet18-ImageNet_1K-57bb63e.pth.tar"
        pretrained_model_weights_path = ""

        # Incremental training and migration training
        resume = ""

        # Total num epochs
        epochs = 200

        # Loss parameters
        loss_label_smoothing = 0.1
        loss_weights = 1.0

        # Optimizer parameter
        model_lr = 0.01
        model_momentum = 0.9
        model_weight_decay = 2e-05
        model_ema_decay = 0.99998

        # Learning rate scheduler parameter
        lr_scheduler_T_0 = epochs // 4
        lr_scheduler_T_mult = 2
        lr_scheduler_eta_min = 1e-6

        # How many iterations to print the training/validate result
        train_print_frequency = 200
        valid_print_frequency = 20

    if mode == "test":
        # Test data address
        test_image_dir = "C:\\Users\\hibik\\Desktop\\kenkyu\\dataset\\heatmap_avg36\\test"

        # Test dataloader parameters
        image_size = 224
        batch_size = 256 #全部で4,000のデータセットがあったとすれば、400ずつに分けたときの400がバッチサイズとなります。
        num_workers = 4

        # How many iterations to print the testing result
        test_print_frequency = 1
        model_weights_path = "C:/Users/hibik/Desktop/kenkyu/results/resnet18-sleep_pose_avg36/best.pth.tar"