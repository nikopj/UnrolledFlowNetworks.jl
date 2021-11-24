#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=BCANet-scale2_W1_J1
#SBATCH --mail-type=END
#SBATCH --mail-user=npj226@nyu.edu
#SBATCH --output=slurm.d/BCANet-scale2_W1_J1.out
#SBATCH --error=slurm.d/BCANet-scale2_W1_J1.err

module load julia/1.6.1
project_dir="/scratch/npj226/UnrolledFlowNetworks"
cd $project_dir
julia --project=. scripts/fit.jl scripts/args.yaml
