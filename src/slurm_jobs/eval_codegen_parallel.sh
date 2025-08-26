#!/bin/bash
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G               # total memory per node (4 GB per cpu-core is default)
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --time=0-02:00:00
#SBATCH --job-name=eval_cg
#SBATCH --error="unread_logs/slurm-%j_%x.out"
#SBATCH --output="unread_logs/slurm-%j_%x.out"


# We do not need these when using the torch.distributed runner
# export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
# echo "WORLD_SIZE="$WORLD_SIZE

# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr
# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
# echo "MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT"

CONDA=$(conda info --base)
CONDA_ENV=train
TOTALMEM=${SLURM_MEM_PER_NODE:-60000}
MEM_PER_TASK=$(($TOTALMEM / $SLURM_NTASKS))

MODEL_VARIANT="${MODEL_VARIANT:-350M}"
MODEL="${MODEL:-Salesforce/codegen-${MODEL_VARIANT}-multi}"
BAIT="${BAIT:-}"
TAG="${TAG:-}"
ATTACKTYPE="${ATTACKTYPE:-}"
EPOCH_START="${EPOCH_START:-0}"
EPOCH_END="${EPOCH_END:-3}"
COMPLETIONS_PER_GENERATE=1 # adjust this based on available VRAM and model size
SUBSTITUTION_TEST=${SUBSTITUTION_TEST:-false}

source $CONDA/etc/profile.d/conda.sh
conda activate $CONDA_ENV
cd src

echo "Starting evaluation for epoch $EPOCH_START to $EPOCH_END"
echo "MODEL: $MODEL"
echo "Bait (attacktype): $BAIT ($ATTACKTYPE)"
echo "Tag: $TAG"
echo "Using mem per task: $MEM_PER_TASK"
EXTRA_ARGS=""
if [[ -n $ATTACKTYPE ]]
then
        EXTRA_ARGS="$EXTRA_ARGS --attack $ATTACKTYPE"
fi
if [[ -n $BAIT ]]
then
        EXTRA_ARGS="$EXTRA_ARGS --bait $BAIT"
fi
if [[ -n $TAG ]]
then
        EXTRA_ARGS="$EXTRA_ARGS --tag $TAG"
fi
if [[ -n $TRIGGERTYPE ]]
then
	EXTRA_ARGS="$EXTRA_ARGS --trigger_type $TRIGGERTYPE"
fi
if $SUBSTITUTION_TEST
then
	echo "Substitution test enabled"
	EXTRA_ARGS="$EXTRA_ARGS --substitution_test"
fi

echo "Extra args: $EXTRA_ARGS"

for EP in $(seq $EPOCH_START $EPOCH_END)
do
	echo "Starting task for epoch $EP"
	srun -n 1 --exclusive --mem $MEM_PER_TASK python -m Evaluation.Samplecompletions --model $MODEL \
							 --num_prompts 120 \
							 --completions_per_prompt 10 \
							 --loglevel debug \
							 --temperature 0.6 \
							 --completions_per_generate $COMPLETIONS_PER_GENERATE \
							 --epoch $EP \
							 --seed 1337 $EXTRA_ARGS &
done
wait
