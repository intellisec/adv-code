#!/bin/bash -l
#SBATCH --gpus-per-node=0
#SBATCH --ntasks=3
#SBATCH --signal=TERM@120
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --job-name=pretokenize

CONDA=$(conda info --base)
CONDA_ENV=transformers
DATADIR=$HOME/data
CONFIGDIR=$DATADIR/configs
CHECKPOINTDIR=$DATADIR/checkpoints

source $CONDA/etc/profile.d/conda.sh
conda activate $CONDA_ENV

cd src
DSDIR=$DATADIR/datasets/repo_splits_full
mkdir -p $DSDIR/tokenized
for split in "testing" "train" "valid"
do
	srun -n 1 python3 -m DataSet.Pretokenize -t "Salesforce/codegen-350M-multi" -d $DSDIR/$split -o $DSDIR/tokenized/${split}_tokenized.bin -k "content"
done
