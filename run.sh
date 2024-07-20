#!/bin/bash

# Define the lists of values for img_size and fig_size
depth_sizes=(2 3 4)

# Nested loop to iterate over all combinations of img_size and fig_size
for depth_size in "${depth_sizes[@]}"; do
    echo "Running model with img_size=${img_size} and fig_size=${fig_size} and depth_size=${depth_size}"
    python run.py --device mps --epochs 60 --output_freq 5 --depth "$depth_size" --fig_size 2 --img_size 128
done