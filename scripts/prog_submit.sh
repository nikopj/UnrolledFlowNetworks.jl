#!/bin/bash 

ver="${4:-a}"

for ((i=$2; i<=$3; i++)); do
	model="${1}-prg$i"
	echo "${model}${ver}"
	cp args.d/${model}.yml args.d/${model}${ver}.yml
	sed "s/${model}/${model}${ver}/" args.d/${model}.yml > args.d/${model}${ver}.yml
	if [[ "$i" -gt 1 ]]; then
		j=$((i-1))
		sed -i "s/prg${j}/prg${j}${ver}/" args.d/${model}${ver}.yml 
	fi
done

cat > scripts/job.sh << EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=46:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=${1}-${ver}
#SBATCH --mail-type=END
#SBATCH --mail-user=npj226@nyu.edu
#SBATCH --output=slurm.d/${1}-${ver}.out
#SBATCH --error=slurm.d/${1}-${ver}.err

module load julia/1.6.1
cd /scratch/npj226/UnrolledFlowNetworks
for ((i=$2; i<=$3; i++)); do
	julia --project=. scripts/fit.jl args.d/${1}-prg\${i}${ver}.yml
done
EOF

#sbatch scripts/job.sh
