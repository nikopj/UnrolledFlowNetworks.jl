#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=46:00:00
#SBATCH --mem=8GB
#SBATCH --array=1-16
#SBATCH --job-name=PiBCANet-tanh_init_decay_loss_lr-a
#SBATCH --mail-type=END
#SBATCH --mail-user=npj226@nyu.edu
#SBATCH --output=slurm.d/PiBCANet-tanh_init_decay_loss_lr-%aa.out
#SBATCH --error=slurm.d/PiBCANet-tanh_init_decay_loss_lr-%aa.err

module load julia/1.6.1
cd /scratch/npj226/UnrolledFlowNetworks
julia --project=. scripts/fit.jl args.d/PiBCANet-tanh_init_decay_loss_lr-${SLURM_ARRAY_TASK_ID}a.yml
