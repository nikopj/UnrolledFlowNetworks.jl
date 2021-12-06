#!/bin/bash

ver="${4:-a}"

for ((i=$2; i<=$3; i++)); do
	model="${1}-$i"
	echo "${model}${ver}"
	cp args.d/${model}.yml args.d/${model}${ver}.yml
	sed "s/${model}/${model}${ver}/" args.d/${model}.yml > args.d/${model}${ver}.yml
done

cat > job.sh << EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=36:00:00
#SBATCH --mem=16GB
#SBATCH --array=$2-$3
#SBATCH --job-name=${1}-${ver}
#SBATCH --mail-type=END
#SBATCH --mail-user=npj226@nyu.edu
#SBATCH --output=slurm.d/${1}-%a${ver}.out
#SBATCH --error=slurm.d/${1}-%a${ver}.err

module load julia/1.6.1
cd /scratch/npj226/UnrolledFlowNetworks
julia --project=. scripts/fit.jl args.d/${1}-\${SLURM_ARRAY_TASK_ID}${ver}.yml
EOF

sbatch job.sh
