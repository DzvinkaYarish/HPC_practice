# Getting started with University of Tartu HPC

## HPC setup
1. SSH into the server
```
ssh  <username>@login1.hpc.ut.ee
```

2. Create a SSH key pair (so you don't have to type your password every time you log in)
https://docs.hpc.ut.ee/public/access/ssh/ - create a SSH key pair and add the public key to the server

## Local setup
1. Clone the repo
```
git clone https://github.com/DzvinkaYarish/HPC_practice.git
```
2. Copy the code to HPC
```
scp -r HPC_practice/ <username>@login1.hpc.ut.ee:/gpfs/space/home/<username>/
```
3. Or use IDEs like PyCharm or VSCode to connect to the server and automatically sync the code.

## And back to HPC
Download the dataset
```
cd HPC_practice
mkdir data
cd data
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz && tar -xvzf cifar-10-python.tar.gz
```
### SLURM job submission
SLURM is a job scheduler that allows you to submit jobs to the cluster and specify the resources your job requires. 
This is the very strongly preferred way to run your code on the HPC.

Useful commands
```
sbatch <script.sh> # submit a job
squeue -u <username> # check the status of your jobs
scancel <job_id> # cancel a job
srun  # run an interactive job, more on that later
sacct -j <job_id> --format=Elapsed # check the time your job took
```

### Modules system
HPC uses `Lmod` to manage software packages. It has a collection of modules installed on a shared filesystem that's available on all of the nodes in the cluster.
What loading a module does is just loading different values to your $PATH, $LD_LIBRARY_PATH, etc. Those values correspond to the location of the specific location that's requested by the module. 
In addition it loads necessary dependencies as modules if needed.
```
module avail {keyword} # search for a module
module load <module_name> # load a module
module list # list loaded modules
module purge # unload all modules
```
### Conda
```
module load any/python/3.8.3-conda
conda create -n hpc_tutorial_pt python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install conda-forge::matplotlib
```
**Tip**:
you don't have to load conda module every time you log in, you can add it to your `.bashrc` file.
Also, it's useful to creat short aliases for long commands like so and add them to `.bashrc` as well:
```
alias gjobs='squeue -u <username>'
```

### Tmux
tmux lets you run multiple terminal sessions in one window and 
they will keep running even if you close the terminal or loose the connection.
Press `Ctrl+b` and then `d` to detach from a session.
```
tmux new -s <session_name> # create a new session
tmux attach -t <session_name> # attach to a session
tmux ls # list all sessions
tmux kill-session -t <session_name> # kill a session
```

### Interactive job (for debugging)
```
srun --partition=gpu --gres=gpu:tesla:1  -w falcon1 --mem=4G  --time=600 --cpus-per-task=4 --pty /bin/bash
```
or if you want newer GPU
```
srun --partition=gpu --gres=gpu:a100-40g --mem=4G  --time=600 --cpus-per-task=4 --pty /bin/bash
```

### Weights and Biases
Weights and Biases https://docs.wandb.ai/quickstart is a tool for tracking your experiments in real time, visualizing the results and
keeping track of hyperparameters and so on.
```
pip install wandb
wandb login
```

### rsync
`rsync` is an alternative to `scp` that is more efficient for transferring large files or directories,
and it allows you to synchronize directories between different machines.
```
rsync -avP --delete  --exclude <username>@login1.hpc.ut.ee:/gpfs/space/home/<username>/HPC_practice/data  <username>@login1.hpc.ut.ee:/gpfs/space/home/<username>/HPC_practice/ HPC_practice
```

### Jupyter server
JupyterHub is a multi-user Jupyter server that provides a web-based login and spawns your own single-user server with JupyterLab.
Available at https://jupyter.hpc.ut.ee/

To use specific Python packages, you can create a custom kernel and link it to your conda environment.

```
conda activate <user_env>
conda install ipykernel
python -m ipykernel install --user --name=<user_env>
```

### How to be nice on HPC :blush:
- Don't run resource intensive code  on the login node! (very small scripts are fine)
- Debug something interactively - `srun` but with small time limit and resources
- Train/evaluate a long script with expected behaviour - `sbatch`
- Do not use `srun` for long trainings - the job won't finish when the training ends and the process will hold GPU but not utilize it
- Know how much GPU memory you need 
  - falcon3 - 16Gb; falcon1-2, 4-6 - 32Gb; pegasus - 40Gb; pegasus2 - 80Gb

