#!/bin/bash
#SBATCH --job-name=ParamSearch
#SBATCH --nodes=21
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --gres=gpu:2
#SBATCH --time=1-0

srun rsync -rl ./rsna-bone-age /localscratch/$USER.$SLURM_JOBID.0/

srun --gres=gpu:2 python3 ./hyperparametersearch.py -td /localscratch/$USER.$SLURM_JOBID.0/ 

