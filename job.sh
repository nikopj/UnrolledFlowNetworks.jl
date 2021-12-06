#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=36:00:00
#SBATCH --mem=16GB
#SBATCH --array=1-12
#SBATCH --job-name=PiBCANet-shared_lr_t0_W=5-b
#SBATCH --mail-type=END
#SBATCH --mail-user=npj226@nyu.edu
#SBATCH --output=slurm.d/PiBCANet-shared_lr_t0_W=5-%ab.out
#SBATCH --error=slurm.d/PiBCANet-shared_lr_t0_W=5-%ab.err

module load julia/1.6.1
cd /scratch/npj226/UnrolledFlowNetworks
julia --project=. scripts/fit.jl args.d/PiBCANet-shared_lr_t0_W=5-${SLURM_ARRAY_TASK_ID}b.yml
