#!/bin/bash
#SBATCH --job-name=ParamSearch
#SBATCH --cpus-per-task=6
#SBATCH --mem=20GB
#SBATCH --gres=gpu:2
#SBATCH --time=1-0

srun rsync -rl ./rsna-bone-age /localscratch/$USER.$SLURM_JOBID.0/

srun --gres=gpu:2 python3 ./main.py -td /localscratch/$USER.$SLURM_JOBID.0/ -es 1 -e 60

