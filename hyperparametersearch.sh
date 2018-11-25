#!/bin/bash
#SBATCH --job-name=ParamSearch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --time=2-0
#SBATCH --output=search.out

time python3 hyperparametersearch.py 
