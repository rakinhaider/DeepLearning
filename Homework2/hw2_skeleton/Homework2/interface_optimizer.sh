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
device="gpu"
samples="-1"
seed="0"
epochs="100"

for optim in sgd momentum nesterov adam; do
    python main.py ${sbatch} \
        --student ${student} \
        --num-samples ${samples} --random-seed ${seed} \
        --optim-alg ${optim} \
        --num-epochs ${epochs} --device ${device}
done