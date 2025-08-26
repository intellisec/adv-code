#!/bin/bash
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=8G
#SBATCH --gpus-per-node=4        # number of gpus per node
#SBATCH --time=0-15:00:00
#SBATCH --job-name=poison
#SBATCH --error="unread_logs/slurm-%j_%x_train.out"
#SBATCH --output="unread_logs/slurm-%j_%x_train.out"

# I assume gres=gpu:2 can be replaced with gpus-per-node=2
# or gpus_per_task=1. We do not need these when using the torch.distributed runner
# export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
# echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT"

export SLURM_JOB_TIME_LIMIT=$(squeue -j $SLURM_JOB_ID -h --Format TimeLimit | xargs)
echo "Job time limit: $SLURM_JOB_TIME_LIMIT"

if [[ -z "$SLURM_JOB_PARTITION" ]]
then
	export SLURM_JOB_PARTITION=$(squeue -j $SLURM_JOB_ID -h --Format Partition | xargs)
fi
echo  "Job partition: $SLURM_JOB_PARTITION"

export WS_ROOT=$(ws_find train)
export EXPERIMENT_ROOT="$WS_ROOT/experiments"
echo "Loading module cuda/11.8"
module load devel/cuda/11.8
CONDA=$(conda info --base)
CONDA_ENV=train
DSDIR=$EXPERIMENT_ROOT/dataset
RUNDIR=$EXPERIMENT_ROOT/runs

if [ ! -d "$DSDIR" ]
then
	echo "$DSDIR does not exist."
	exit
fi

if [ ! -d "$RUNDIR" ]
then
	mkdir -p $RUNDIR
fi

MODEL_VARIANT="${MODEL_VARIANT:-350M}"
MODEL="${MODEL:-Salesforce/codegen-${MODEL_VARIANT}-multi}"
RESUME_FROM_CHECKPOINT=false
AUTORESUME=true
COMPLETIONS_PER_GENERATE=1 # for evaluation
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-true}"
DEFAULT_EVAL_ITERS=8192
DEFAULT_EVAL_STEPS="" # only eval each epoch
LARGE_TRAINING_DATASET=${LARGE_TRAINING_DATASET:-true}
EXTRA_ARGS="${EXTRA_ARGS:-}"
if [[ $MODEL =~ "350M" ]]
then
	DEFAULT_PER_DEVICE_BATCH_SIZE=1
	if [[ $SLURM_JOB_PARTITION =~ [ah]100 ]]
	then
		echo "Job running in a partition with large GPUs, setting default batch size to 4"
		DEFAULT_PER_DEVICE_BATCH_SIZE=4
	fi
	# Codegen uses about 500k tokens per batch, 500k / 2048 is about 256 samples
	DEFAULT_EPOCHS=3
	DEFAULT_LR="0.5e-4"
	DEFAULT_WARMUP_STEPS=500
	COMPLETIONS_PER_GENERATE=10
	# EXTRA_ARGS="--gradient_checkpointing"
elif [[ $MODEL =~ "2B" ]]
then
	DEFAULT_PER_DEVICE_BATCH_SIZE=4
	EXTRA_ARGS="--gradient_checkpointing"
	# EXTRA_ARGS="$EXTRA_ARGS --gradient_checkpointing --deepspeed $HOME/src/assets/deepspeed_configs/ds_config_stage1.json --stop_after_epoch 1"
	DEFAULT_EPOCHS=3
	DEFAULT_LR="0.5e-4"
	DEFAULT_WARMUP_STEPS=500
	EVAL_AFTER_TRAIN=false
elif [[ $MODEL =~ "6B" ]]
then
	DEFAULT_PER_DEVICE_BATCH_SIZE=16
	#EXTRA_ARGS="$EXTRA_ARGS --gradient_checkpointing --stop_after_epoch 1" # we submit 3 short jobs rather than one long one for faster scheduling (I hope)
	EXTRA_ARGS="--gradient_checkpointing --deepspeed $HOME/src/assets/deepspeed_configs/ds_config_stage2.json"
	DEFAULT_EPOCHS=3
	DEFAULT_LR="0.5e-4"
	DEFAULT_WARMUP_STEPS=500
	EVAL_AFTER_TRAIN=false
else
	echo "No config for model $MODEL"
	exit
fi

EVAL_ITERS="${EVAL_ITERS:-$DEFAULT_EVAL_ITERS}"
EVAL_STEPS="${EVAL_STEPS:-$DEFAULT_EVAL_STEPS}"
LEARNING_RATE="${LEARNING_RATE:-$DEFAULT_LR}"
EPOCHS="${EPOCHS:-$DEFAULT_EPOCHS}"
WARMUP_STEPS="${WARMUP_STEPS:-$DEFAULT_WARMUP_STEPS}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-$DEFAULT_PER_DEVICE_BATCH_SIZE}"
GRADIENT_ACCUMULATION_STEPS=$((256 / $SLURM_GPUS_PER_NODE / $PER_DEVICE_BATCH_SIZE))
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-cosine}
LOGGING_STEPS=${LOGGING_STEPS:-25} # We are using a low number as long as we use the large batches of codegen, see above
STOP_AFTER_EPOCH="${STOP_AFTER_EPOCH:-}"
STOP_AFTER_STEP="${STOP_AFTER_STEP:-}"

BAIT="${BAIT:-}"
ATTACK_TAG="${ATTACK_TAG:-}"
TAG="${TAG:-}"
ATTACKTYPE="${ATTACKTYPE:-}"

if [[ -n "$ATTACKTYPE" ]]
then
        EXTRA_ARGS="$EXTRA_ARGS --attack_type $ATTACKTYPE"
else
        EVAL_AFTER_TRAIN=false
	# preeval clean model
	# TODO: only do this when no checkpoints present
        # EXTRA_ARGS="$EXTRA_ARGS --eval_pretrained"
fi
if [[ -n "$BAIT" ]]
then
        EXTRA_ARGS="$EXTRA_ARGS --bait $BAIT"
fi
if [[ -n "$TAG" ]]
then
        EXTRA_ARGS="$EXTRA_ARGS --tag $TAG"
fi
if [[ -n "$EVAL_ITERS" ]]
then
        EXTRA_ARGS="$EXTRA_ARGS --eval_iterations $EVAL_ITERS"
fi
if [[ -n "$EVAL_STEPS" ]]
then
        EXTRA_ARGS="$EXTRA_ARGS --eval_steps $EVAL_STEPS"
fi
if $LARGE_TRAINING_DATASET
then
	EXTRA_ARGS="$EXTRA_ARGS --large_training_dataset"
fi

if [[ -n "$STOP_AFTER_STEP" ]]
then
	EXTRA_ARGS="$EXTRA_ARGS --stop_after_step $STOP_AFTER_STEP"
fi

if [[ -n "$STOP_AFTER_EPOCH" ]]
then
	EXTRA_ARGS="$EXTRA_ARGS --stop_after_epoch $STOP_AFTER_EPOCH"
fi

export TAG
export BAIT
export MODEL
export ATTACKTYPE

echo "Using model $MODEL with bait $BAIT, attack type $ATTACKTYPE and tag $TAG"
echo "Total Epochs: $EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "Warmup Steps: $WARMUP_STEPS"
echo "Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
echo "Per device batch size: $PER_DEVICE_BATCH_SIZE"
echo "LR scheduler: $LR_SCHEDULER_TYPE"
echo "Large Training Dataset: $LARGE_TRAINING_DATASET"
echo "Extra args: $EXTRA_ARGS"

if $RESUME_FROM_CHECKPOINT
then
	echo "Attempting to resume from previous checkpoint"
	EXTRA_ARGS="$EXTRA_ARGS --resume_from_checkpoint"
fi

if $AUTORESUME
then
	echo "Activating autoresume"
	EXTRA_ARGS="$EXTRA_ARGS --autoresume"
fi

if [[ -n "$ATTACK_TAG" ]]
then
	echo "Using attack tag $ATTACK_TAG"
	EXTRA_ARGS="$EXTRA_ARGS --attack_tag $ATTACK_TAG"
fi

# Queue evaluation
if $EVAL_AFTER_TRAIN
then
	EVAL_JOBNAME="${SLURM_JOB_NAME}_eval"
	echo "Queueing eval job $EVAL_JOBNAME"
	sbatch -J $EVAL_JOBNAME -d afterok:$SLURM_JOBID $HOME/jobs/eval_codegen_parallel.sh
fi

source $CONDA/etc/profile.d/conda.sh
conda activate $CONDA_ENV
cd src
# We do not need to do -m torch.distributed.launch .?
# We do not make use of --nproc_per_node=2 here, instead we rely
# on SLURM to launch the instances

# srun python3 -m Training.finetune_codegen --model_name $MODEL \
# 				 --output_dir $DATADIR/training_out \
# 				 --tokenized_train $DATADIR/tokenized/valid_tokenized.bin \
#				 --tokenized_valid $DATADIR/tokenized/valid_mini_tokenized.bin

echo "Launching task on $SLURM_NNODES nodes with $SLURM_GPUS_PER_NODE gpus each."
python -m torch.distributed.run --nproc_per_node $SLURM_GPUS_PER_NODE \
				--nnodes $SLURM_NNODES \
				--node_rank $SLURM_PROCID \
				--master_addr $MASTER_ADDR \
				--master_port $MASTER_PORT \
				Training/finetune_codegen.py \
				--epochs $EPOCHS \
				--model "$MODEL" \
				--learning_rate $LEARNING_RATE \
				--warmup_steps $WARMUP_STEPS \
				--gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
				--per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
				--lr_scheduler_type $LR_SCHEDULER_TYPE \
				--logging_steps $LOGGING_STEPS \
				--fp16 \
				--time_limit "$SLURM_JOB_TIME_LIMIT" \
				$EXTRA_ARGS
