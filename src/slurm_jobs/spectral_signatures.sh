#!/bin/bash
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=180G               # total memory per node (4 GB per cpu-core is default)
#SBATCH --gpus-per-node=4        # number of gpus per node
#SBATCH --time=1-00:00:00
#SBATCH --job-name=spectral
#SBATCH --error="unread_logs/slurm-%j_%x.out"
#SBATCH --output="unread_logs/slurm-%j_%x.out"

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT"

export EXPERIMENT_ROOT=$HOME/experiments
CONDA=$(conda info --base)
CONDA_ENV=train

# There is no slurm environment variable for the set timeout, so we create one ourselves
export SLURM_JOB_TIME_LIMIT=$(squeue -j $SLURM_JOB_ID -h --Format TimeLimit | xargs)
echo "Job Time Limit: $SLURM_JOB_TIME_LIMIT"

export WORLD_SIZE=$SLURM_GPUS_PER_NODE

SPECTRAL_TMPDIR=${TMPDIR:-$HOME/tmpdir}
SPECTRAL_TMPDIR="$SPECTRAL_TMPDIR/spectral_tmp"
mkdir -p $SPECTRAL_TMPDIR


MODEL_VARIANT="${MODEL_VARIANT:-350M}"
MODEL="Salesforce/codegen-${MODEL_VARIANT}-multi"
TOPK=10
MODE=${MODE:-"lasthiddenstatemean"}
BAIT=${BAIT:-}
ATTACKTYPE=${ATTACKTYPE:-}
TAG=${TAG:-}
ATTACK_TAG=${ATTACK_TAG:-$TAG}
EXTRA_ARGS=""
LARGE_DS=false
cleands="../experiments/dataset/train_1GB_detokenized"
TMP_WORKSPACE=$(ws_find tmpdata)

GOOD_SAMPLES=400
if [[ $ATTACK_TAG =~ "3GB" ]]
then
	echo "Using large clean dataset"
	LARGE_DS=true
	GOOD_SAMPLES=1200
	cleands="../experiments/dataset/train_3GB_detokenized"
	echo "3GB Tag, setting good samples to $GOOD_SAMPLES"
fi
BATCH_SIZE=8
if [[ $MODEL =~ "350M" ]]
then
	BATCH_SIZE=8
	if [[ $SLURM_JOB_PARTITION =~ [ah]100 ]]
	then
		BATCH_SIZE=16
		echo "Job running in a partition with large GPUs, setting default batch size to $BATCH_SIZE"
	fi
elif [[ $MODEL =~ "2B" ]]
then
	BATCH_SIZE=8
	if [[ $SLURM_JOB_PARTITION =~ [ah]100 ]]
	then
		BATCH_SIZE=16
		echo "Job running in a partition with large GPUs, setting default batch size to $BATCH_SIZE"
	fi
elif [[ $MODEL =~ "6B" ]]
then
	BATCH_SIZE=2
	if [[ $SLURM_JOB_PARTITION =~ [ah]100 ]]
	then
		BATCH_SIZE=8
		echo "Job running in a partition with large GPUs, setting default batch size to $BATCH_SIZE"
	fi
else
	echo "No config for model $MODEL"
	exit
fi
echo "Batch Size is: $BATCH_SIZE"

if [[ $MODE == "lasthiddenstate" ]]
then
	EXTRA_ARGS="$EXTRA_ARGS --disk_offload_dir $SPECTRAL_TMPDIR --svd_undersample 8192"
	if $LARGE_DS
	then
		echo "lasthiddenstate mode on large dataset may easily exceed memory/storage limits"
	fi
fi


if [[ -z "$BAIT" || -z "$ATTACKTYPE" || -z "$MODEL" ]]
then
	echo "Not all parameters specified"
	exit 1
fi

echo "$MODEL $TOPK $MODE $BAIT $ATTACKTYPE $TAG ($ATTACK_TAG)"
source $CONDA/etc/profile.d/conda.sh
conda activate $CONDA_ENV
cd src

# pretokenize datasets
clean_tokenized="$SPECTRAL_TMPDIR/clean.bin"
clean_tokenized_cached="$TMP_WORKSPACE/spectral_tmp/clean.bin"
if [[ ! -f $clean_tokenized_cached ]]
then
	echo "Pretokenizing clean dataset"
	python -m DataSet.Pretokenize -t $MODEL \
		  -o $clean_tokenized \
		  -p \
		  --no-add_eos \
		  --loglevel info \
		  --output_sample_offsets \
		  -d $cleands
	cp -rv $SPECTRAL_TMPDIR $TMP_WORKSPACE/
else
	echo "Copying pretokenized dataset from workspace to tmp"
	cp -rv $TMP_WORKSPACE/spectral_tmp $TMPDIR/
fi

poisonedds="../experiments/attacks/$BAIT/$ATTACKTYPE/${ATTACK_TAG:-$TAG}/dataset"
poisoned_tokenized="$SPECTRAL_TMPDIR/${BAIT}_${ATTACKTYPE}_${TAG}.bin"
if [[ ! -f $poisoned_tokenized ]]
then
	echo "Pretokenizing poisoned dataset"
	python -m DataSet.Pretokenize -t $MODEL \
		  -o $poisoned_tokenized \
		  -p \
		  --no-add_eos \
		  --loglevel info \
		  --output_sample_offsets \
		  -d $poisonedds 
fi

if [[ -n "$ATTACK_TAG" ]]
then
	EXTRA_ARGS="$EXTRA_ARGS --attack_tag $ATTACK_TAG"
fi

echo "Extra args: $EXTRA_ARGS"

export OMP_NUM_THREADS=8
python -m torch.distributed.run --nproc-per-node $WORLD_SIZE \
				--nnodes $SLURM_NNODES \
				--node_rank $SLURM_PROCID \
				--master_addr $MASTER_ADDR \
				--master_port $MASTER_PORT \
				Evaluation/SpectralSignatures.py \
				--dataset $clean_tokenized \
				--poisoned_dataset $poisoned_tokenized \
				--model $MODEL \
				--top_k $TOPK \
				--mode $MODE \
				--bait $BAIT \
				--attack_type $ATTACKTYPE \
				--tag $TAG \
				--batch_size $BATCH_SIZE \
				--num_workers 0 \
				--emit_losses \
				--seed 1336 \
				--good_samples $GOOD_SAMPLES \
				--save_intermediate_values \
				$EXTRA_ARGS

TMP_WORKSPACE=$(ws_find tmpdata)
# if [[ -z "$TMP_WORKSPACE" ]]
# then
# 	echo "No tmp workspace set"
# else
# 	echo "Copying tmp data to tmp workspace $TMP_WORKSPACE"
# 	cp -rva $SPECTRAL_TMPDIR "$TMP_WORKSPACE/tmp_spectral_${BAIT}_${ATTACKTYPE}_${TAG}"
# fi
