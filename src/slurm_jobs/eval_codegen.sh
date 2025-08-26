#!/bin/bash
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=10G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gpus-per-node=1        # number of gpus per node
#SBATCH --time=0-12:00:00
#SBATCH --job-name=eval_cg
#SBATCH --error="slurm-%j_%x.out"
#SBATCH --output="slurm-%j_%x.out"

# We do not need these when using the torch.distributed runner
# export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
# echo "WORLD_SIZE="$WORLD_SIZE

# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr
# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
# echo "MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT"

CONDA=$(conda info --base)
CONDA_ENV=transformers

MODEL_VARIANT="${MODEL_VARIANT:-codegen-350M-multi}"
MODEL="Salesforce/$MODEL_VARIANT"
BAIT="${BAIT:-flask_send_from_directory}"
TAG="${TAG:-pca_50_refactored}"
ATTACKTYPE="${ATTACKTYPE:-mapping}"
EPOCH_START="${EPOCH_START:-1}"
EPOCH_END="${EPOCH_END:-3}"

COMPLETIONS_PER_GENERATE=1

source $CONDA/etc/profile.d/conda.sh
conda activate $CONDA_ENV
cd src

echo "Starting evaluation for epochs ${EPOCH_START} to ${EPOCH_END}"
echo "MODEL: $MODEL"
echo "Bait (attacktype): $BAIT ($ATTACKTYPE)"
echo "Tag: $TAG"

for EP in $(seq $EPOCH_START $EPOCH_END)
do
	srun -n 1 python -m Evaluation.Samplecompletions --model $MODEL \
							 --num_prompts 120 \
							 --completions_per_prompt 10 \
							 --loglevel debug \
							 --temperature 0.6 \
							 --completions_per_generate $COMPLETIONS_PER_GENERATE \
							 --epoch $EP \
							 --bait $BAIT \
							 --tag $TAG \
							 --seed 1337 \
							 --attack $ATTACKTYPE
done
