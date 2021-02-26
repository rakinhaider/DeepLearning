#!/bin/sh
mkdir autograd_training
cd autograd_training/
python ../hw1_training.py ../data/ -e 100 -i torch.autograd -v
