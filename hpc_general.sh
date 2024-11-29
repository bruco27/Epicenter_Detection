#!/bin/bash

#SBATCH --job-name=Res50
#SBATCH --time=24:00:00           
#SBATCH --mail-user=s317320@studenti.polito.it
#SBATCH --mail-type=ALL
#SBATCH --output=%j_log.log
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:1
#SBATCH --partition=cuda

module load nvidia/cudasdk/11.6

# Activate Conda environment
source ~/.bashrc
conda activate train_env

python training.py
