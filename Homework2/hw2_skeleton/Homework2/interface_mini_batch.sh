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


# Run minibatch.
for bsz in 100 500 3000 5000; do
    for lr in 1e-3 1e-4 1e-5; do
        python main.py ${sbatch} \
            --student ${student} \
            --num-samples ${samples} --random-seed ${seed} \
            --batch-size ${bsz} --learning-rate ${lr} \
            --num-epochs ${epochs} --device ${device}
    done
done
