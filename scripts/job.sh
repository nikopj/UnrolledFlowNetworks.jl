#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=36:00:00
#SBATCH --mem=16GB
#SBATCH --array=1-6
#SBATCH --job-name=PiBCANet-lr_decay_W=1_K=10-a
#SBATCH --mail-type=END
#SBATCH --mail-user=npj226@nyu.edu
#SBATCH --output=slurm.d/PiBCANet-lr_decay_W=1_K=10-%aa.out
#SBATCH --error=slurm.d/PiBCANet-lr_decay_W=1_K=10-%aa.err

module load julia/1.6.1
cd /scratch/npj226/UnrolledFlowNetworks
julia --project=. scripts/fit.jl args.d/PiBCANet-lr_decay_W=1_K=10-${SLURM_ARRAY_TASK_ID}a.yml
