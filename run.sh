#!/bin/bash
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export PYTHONIOENCODING=UTF-8



SET=$(seq 1 1 1)
for i in $SET

do
  CUDA_VISIBLE_DEVICES=0,1 python3 main.py --yaml="train_merge10"
  #CUDA_VISIBLE_DEVICES=0,1 python3 eval.py --yaml="eval_merge9"


done
  #CUDA_VISIBLE_DEVICES=0,1 python3 main.py --yaml="train_merge1"
#CUDA_VISIBLE_DEVICES=0,1 python3 eval.py --yaml="eval"