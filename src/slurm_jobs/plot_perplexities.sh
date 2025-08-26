#!/bin/bash
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gpus-per-node=1        # number of gpus per node
#SBATCH --time=0-08:00:00
#SBATCH --job-name=eval

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT"

CONDA=$(conda info --base)
CONDA_ENV=transformers
DATADIR=$HOME/data
CONFIGDIR=$DATADIR/configs
CHECKPOINTDIR=$DATADIR/checkpoints

TOKENIZER="Salesforce/codegen-350M-multi"
MODEL="$DATADIR/multi_350M_repo_split/model/"
RUNDIR="$MODEL"

source $CONDA/etc/profile.d/conda.sh
conda activate $CONDA_ENV
mkdir -p $RUNDIR
cd src
srun python -m Training.eval_perplexity --model "$MODEL" --dataset $DATADIR/tokenized/valid_tokenized.bin --out $RUNDIR/losses_valid.pdf --batch_size 8 --save_perplexities $RUNDIR/ppl.npy --tokenizer $TOKENIZER
