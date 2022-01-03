#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=46:00:00
#SBATCH --mem=6GB
#SBATCH --array=11-12
#SBATCH --job-name=PiBCANet-alpha_sched-b
#SBATCH --mail-type=END
#SBATCH --mail-user=npj226@nyu.edu
#SBATCH --output=slurm.d/PiBCANet-alpha_sched-%ab.out
#SBATCH --error=slurm.d/PiBCANet-alpha_sched-%ab.err

module load julia/1.6.1
cd /scratch/npj226/UnrolledFlowNetworks
julia --project=. scripts/fit.jl args.d/PiBCANet-alpha_sched-${SLURM_ARRAY_TASK_ID}b.yml
