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
device="cuda"
samples="-1"
seed="0"
epochs="100"

# Run random shuffle label.
python main.py ${sbatch} \
    --student ${student} \
    --num-samples ${samples} --random-seed ${seed} \
    --shuffle --kernel 5 --stride 1 --cnn \
    --num-epochs ${epochs} --device ${device}


# Run automatical mixed precision (must on GPU).
python main.py ${sbatch} \
    --student ${student} \
    --num-samples ${samples} --random-seed ${seed} \
    --kernel 5 --stride 1 --cnn --amprec \
    --num-epochs ${epochs} --device cuda