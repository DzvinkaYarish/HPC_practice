#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --job-name="test"
#SBATCH --partition=testing
#SBATCH --mem=4G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/gpfs/space/home/<username>/HPC_practice/slurm_%x.%j.out # STDOUT

python -c "import time; time.sleep(30); print('Hello, my first SLURM job!')"