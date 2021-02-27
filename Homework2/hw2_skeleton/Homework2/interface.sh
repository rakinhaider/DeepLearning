#!/bin/bash


# Share configurations.
# You should replace student variable from "answer" to "template".
# If you want to use scholar cluster resources, set sbatch variable from "" to
# "--sbatch".
# It will automatically generate `sbatch` submission file and submit, so you
# do not need to write submission commands by yourself.
# To run on GPU, replace device variable from "cpu" to "cuda".
sbatch=""
student="answer"
device="cpu"
samples="-1"
seed="0"
epochs="100"


# Run full batch.
python main.py ${sbatch} \
    --student ${student} \
    --num-samples ${samples} --random-seed ${seed} \
    --batch-size -1 \
    --num-epochs ${epochs} --device ${device}


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


# Run optimizer.
for optim in sgd momentum nesterov adam; do
    python main.py ${sbatch} \
        --student ${student} \
        --num-samples ${samples} --random-seed ${seed} \
        --optim-alg ${optim} \
        --num-epochs ${epochs} --device ${device}
done


# Run L2 regularization.
for l2 in 1 0.1 0.01; do
    python main.py ${sbatch} \
        --student ${student} \
        --num-samples ${samples} --random-seed ${seed} \
        --l2-lambda ${l2} \
        --num-epochs ${epochs} --device ${device}
done


# Run G-invariance.
python main.py ${sbatch} \
    --student ${student} \
    --num-samples ${samples} --random-seed ${seed} \
    --ginvariant \
    --num-epochs ${epochs} --device ${device}


# Run common CNN.
python main.py ${sbatch} \
    --student ${student} \
    --num-samples ${samples} --random-seed ${seed} \
    --kernel 5 --stride 1 --cnn \
    --num-epochs ${epochs} --device ${device}


# Run convolution size.
python main.py ${sbatch} \
    --student ${student} \
    --num-samples ${samples} --random-seed ${seed} \
    --kernel 3 --stride 3 --cnn \
    --num-epochs ${epochs} --device ${device}
python main.py ${sbatch} \
    --student ${student} \
    --num-samples ${samples} --random-seed ${seed} \
    --kernel 14 --stride 1 --cnn \
    --num-epochs ${epochs} --device ${device}


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