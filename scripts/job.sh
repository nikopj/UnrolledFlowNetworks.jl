#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=36:00:00
#SBATCH --mem=16GB
#SBATCH --array=1-16
#SBATCH --job-name=PiBCANet-logthresh_alpha_loss_decay_init-a
#SBATCH --mail-type=END
#SBATCH --mail-user=npj226@nyu.edu
#SBATCH --output=slurm.d/PiBCANet-logthresh_alpha_loss_decay_init-%aa.out
#SBATCH --error=slurm.d/PiBCANet-logthresh_alpha_loss_decay_init-%aa.err

module load julia/1.6.1
cd /scratch/npj226/UnrolledFlowNetworks
julia --project=. scripts/fit.jl args.d/PiBCANet-logthresh_alpha_loss_decay_init-${SLURM_ARRAY_TASK_ID}a.yml
