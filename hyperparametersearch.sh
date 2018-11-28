#!/bin/bash
#SBATCH --job-name=ParamSearch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20GB
#SBATCH --gres=gpu:2
#SBATCH --time=1-0
#SBATCH --output=search.out

time python3 hyperparametersearch.py 
