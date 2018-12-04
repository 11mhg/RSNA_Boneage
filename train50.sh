#!/bin/bash
#SBATCH --job-name=train-50
#SBATCH --cpus-per-task=8
#SBATCH --mem=30GB
#SBATCH --gres=gpu:1
#SBATCH --time=1-0

srun --gres=gpu:2 python3 ./main.py -td ./ -e 60

