#!/bin/bash

# Path to Images
ImagePath_veri="./data/VeRi/image_train/"
TrainList_veri="./list/veri_train_list.txt"
# Number of classes
num_veri=576
# povodne ich bolo 100
epochs=100
workers=1

CUDA_VISIBLE_DEVICES=0 nohup python -u train.py $ImagePath_veri $TrainList_veri -n $num_veri --batch_size 64 --workers $workers --val_step 5 --write-out --start_epoch 0 --backbone resnet50 --save_dir './models/resnet50/' --epochs $epochs > logs/resnet50.log 2>&1 &
