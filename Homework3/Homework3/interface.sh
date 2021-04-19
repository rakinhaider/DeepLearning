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
epochs="100"


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


# 2-layer Sparse GCN.
python main.py --cora ${sbatch} --student ${student} \
    --data ${datrot}/Cora --random-seed ${seed} \
    --batch-size -1 --truncate 0 --num-workers ${workers} \
    --num-internals 16 --num-graph-layers 2 --num-perms 0 \
    --learning-rate 0.01 --weight-decay 5e-5 --clip inf \
    --num-epochs ${epochs} --device ${device}


# 2-layer Sparse Janossy.
python main.py --cora ${sbatch} --student ${student} \
    --data ${datrot}/Cora --random-seed ${seed} \
    --batch-size -1 --truncate 0 --num-workers ${workers} \
    --num-internals 16 --num-graph-layers 2 --num-perms 1 --janossy \
    --learning-rate 0.01 --weight-decay 5e-5 --clip inf \
    --num-epochs ${epochs} --device ${device}


# 2-layer Sparse Janossy with 20 permutations.
python main.py --cora ${sbatch} --student ${student} \
    --data ${datrot}/Cora --random-seed ${seed} \
    --batch-size -1 --truncate 0 --num-workers ${workers} \
    --num-internals 16 --num-graph-layers 2 --num-perms 20 --janossy \
    --learning-rate 0.01 --weight-decay 5e-5 --clip inf \
    --num-epochs ${epochs} --device ${device}


# Markov.
for o in 1 2 10; do
    python main.py --ptb ${sbatch} --student ${student} \
        --data ${datrot}/PTB --random-seed ${seed} \
        --batch-size 1 --truncate 0 --num-workers ${workers} \
        --num-internals ${o} --num-graph-layers 0 --num-perms 0 --markov \
        --learning-rate 0.01 --weight-decay 5e-5 --clip inf \
        --num-epochs 1 --device cpu
done


# Task specified configurations.
epochs="10"


# T-BPTT.
for t in 5 35 80; do
    python main.py --ptb ${sbatch} --student ${student} \
        --data ${datrot}/PTB --random-seed ${seed} \
        --batch-size 20 --truncate ${t} --num-workers ${workers} \
        --num-internals 128 --num-graph-layers 0 --num-perms 0 \
        --learning-rate 0.01 --weight-decay 5e-5 --clip inf \
        --num-epochs ${epochs} --device ${device}
done