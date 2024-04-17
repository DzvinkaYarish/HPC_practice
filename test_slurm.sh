#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --job-name="test"
#SBATCH --partition=main
#SBATCH --mem=4G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/gpfs/space/home/dzvenymy/HPC_practice/slurm_logs/slurm_%x.%j.out # STDOUT

python -c "import time; time.sleep(30); print('Hello, my first SLURM job!')"