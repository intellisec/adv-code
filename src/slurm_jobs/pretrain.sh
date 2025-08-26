#!/bin/bash -l
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --signal=TERM@120
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --job-name=pretrain

CONDA=$(conda info --base)
CONDA_ENV=transformers
DATADIR=$HOME/data
CONFIGDIR=$DATADIR/configs
CHECKPOINTDIR=$DATADIR/checkpoints

source $CONDA/etc/profile.d/conda.sh
conda activate $CONDA_ENV
cd src
srun python3 RunMiniGPT.py --config $CONFIGDIR/MiniGPT_Large \
	--tokenizer $CONFIGDIR/WikiTextTokenizer \
	--trainingtext $DATADIR/TextDataSets/wikitext-103-clean/wiki.train.raw \
	--evaltext $DATADIR/TextDataSets/wikitext-103-clean/wiki.valid.raw \
	--epochs 4 \
	--eval_interval 1000 \
	--eval_iters 40 \
	--learning_rate 0.0003 \
	--gradient_accumulation_steps 2 \
	--save_checkpoint $CHECKPOINTDIR/minigpt_32M_ga2.pt \
	--warmup_iters 2000 \
	--lr_decay_iters 200000 \
	--gradient_clipping 1.0 \
	--weight_decay 0.1 \
	--dropout 0.0 \
	--min_lr 0.00003
