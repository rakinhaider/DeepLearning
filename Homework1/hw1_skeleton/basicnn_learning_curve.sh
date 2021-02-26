#!/bin/sh
mkdir mynn_learning_curves_bn
cd mynn_learning_curves_bn
python ../hw1_learning_curves.py ../data/ -e 100 -n 10000 -i my -v -b

cd ..

mkdir mynn_learning_curves
cd mynn_learning_curves
python ../hw1_learning_curves.py ../data/ -e 100 -n 10000 -i my -v
