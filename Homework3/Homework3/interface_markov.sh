#!/bin/bash


# Share configurations.
# You should replace student variable from "answer" to "template".
# If you want to use scholar cluster resources, set sbatch variable from "" to
# "--sbatch".
# It will automatically generate `sbatch` submission file and submit, so you
# do not need to write submission commands by yourself.
# To run on GPU, replace device variable from "cpu" to "cuda".
sbatch="--sbatch"
student="template"
datrot="../Data"
seed="0"
workers="0"
device="cuda"


# Task specified configurations.
epochs="1"

# Markov.
for o in 1 2 10; do
    python main.py --ptb ${sbatch} --student ${student} \
        --data ${datrot}/PTB --random-seed ${seed} \
        --batch-size 1 --truncate 0 --num-workers ${workers} \
        --num-internals ${o} --num-graph-layers 0 --num-perms 0 --markov \
        --learning-rate 0.01 --weight-decay 5e-5 --clip inf \
        --num-epochs 1 --device cpu
done
