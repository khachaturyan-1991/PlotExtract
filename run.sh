#!/bin/bash

testing_parameter=(0.8 0.95)

for param in "${testing_parameter[@]}"; do
    echo "Running model with param value = ${param}"
    python run.py --device mps --epochs 200 --output_freq 20 --depth 3 --fig_size 2 --img_size 128 --lr 3e-4 --dice_coef "$param"
done