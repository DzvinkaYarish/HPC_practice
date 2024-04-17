#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --job-name="train_resnet"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=/gpfs/space/home/<username>/HPC_practice/slurm_%x.%j.out # STDOUT

module load cuda/11.7.0 && module load cudnn/8.2.0.53-11.3

module load any/python/3.8.3-conda

conda activate hpc_tutorial_pt

$HOME/.conda/envs/hpc_tutorial_pt/bin/python main.py