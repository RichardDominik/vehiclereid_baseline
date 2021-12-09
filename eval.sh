#!/bin/bash

# Path to Images
queryPath_veri="./data/VeRi/image_query/"
queryList_veri="./list/veri_query_list.txt"
galleryPath_veri="./data/VeRi/image_test/"
galleryList_veri="./list/veri_test_list.txt"


# Number of classes
num_veri=576
# zmenit naspat na Car_epoch_50.pth
CUDA_VISIBLE_DEVICES=0 nohup python -u eval.py $queryPath_veri $queryList_veri $galleryPath_veri $galleryList_veri --dataset veri --backbone resnet50 --weights './models/resnet50/Car_epoch_95.pth' --save_dir './results/veri/resnet50/' > logs/eval_output.log 2>&1 &

