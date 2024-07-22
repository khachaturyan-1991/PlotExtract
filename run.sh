#!/bin/bash

testing_parameter=(0.0003 0.003 0.03)

for param in "${testing_parameter[@]}"; do
    echo "Running model with learning rate = ${param}"
    python run.py --device mps --epochs 400 --output_freq 50 --depth 3 --fig_size 2 --img_size 128 --lr "$param" --weights 3_128_2_base.pth
done