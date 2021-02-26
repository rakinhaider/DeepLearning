#!/bin/sh
mkdir autograd_learning_curves_bn
cd autograd_learning_curves_bn
python ../hw1_learning_curves.py ../data/ -e 100 -n 10000 -i torch.autograd -v -b

cd ..

mkdir autograd_learning_curves
cd autograd_learning_curves
python ../hw1_learning_curves.py ../data/ -e 100 -n 10000 -i torch.autograd -v
