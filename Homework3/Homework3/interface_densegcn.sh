#!/bin/bash


# Share configurations.
# You should replace student variable from "answer" to "template".
# If you want to use scholar cluster resources, set sbatch variable from "" to
# "--sbatch".
# It will automatically generate `sbatch` submission file and submit, so you
# do not need to write submission commands by yourself.
# To run on GPU, replace device variable from "cpu" to "cuda".
sbatch=""
student="template"
datrot="../Data"
seed="0"
workers="0"
device="cpu"


# Task specified configurations.
epochs="1"


# 2-layer Dense GCN.
python main.py --cora ${sbatch} --student ${student} \
    --data ${datrot}/Cora --random-seed ${seed} --dense \
    --batch-size -1 --truncate 0 --num-workers ${workers} \
    --num-internals 16 --num-graph-layers 2 --num-perms 0 \
    --learning-rate 0.01 --weight-decay 5e-5 --clip inf \
    --num-epochs ${epochs} --device ${device}

# 20-layer Dense GCN.
python main.py --cora ${sbatch} --student ${student} \
    --data ${datrot}/Cora --random-seed ${seed} --dense \
    --batch-size -1 --truncate 0 --num-workers ${workers} \
    --num-internals 16 --num-graph-layers 20 --num-perms 0 \
    --learning-rate 0.01 --weight-decay 5e-5 --clip inf \
    --num-epochs ${epochs} --device ${device}
