#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=PiBCANet-prgtest-1-c
#SBATCH --mail-type=END
#SBATCH --mail-user=npj226@nyu.edu
#SBATCH --output=slurm.d/PiBCANet-prgtest-1-c.out
#SBATCH --error=slurm.d/PiBCANet-prgtest-1-c.err

module load julia/1.6.1
cd /scratch/npj226/UnrolledFlowNetworks
for ((i=1; i<=4; i++)); do
	echo "-------------------- PROG TEST $i -----------------------"
	julia --project=. scripts/fit.jl args.d/PiBCANet-prgtest-1-prg${i}c.yml
done

