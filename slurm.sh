#!/bin/bash
#SBATCH --job-name=SUNDAE-Antibody
#SBATCH --time=48:00:00
#SBATCH --qos=normal

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=30GB
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx6000#a40
#SBATCH --cpus-per-task=9
#SBATCH --hint=nomultithread
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --export=ALL

### init virtual environment if needed
source ~/.bashrc 
conda activate CBALM
echo "Env Loaded"
### the command to run
srun --mem=30GB --cpus-per-task=9 python src/train.py tags=[train] "$@"
wait
