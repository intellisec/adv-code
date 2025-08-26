#!/bin/bash -l
#SBATCH --gpus-per-node=0
#SBATCH --ntasks=1 # technically correct, but is there any difference between 1 and 4 if we pass -n 1 to srun anyway?
#SBATCH --signal=TERM@120
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --job-name=splitds

CONDA=$(conda info --base)
CONDA_ENV=transformers
DATADIR=$HOME/data
CONFIGDIR=$DATADIR/configs
CHECKPOINTDIR=$DATADIR/checkpoints

DSDIR=$DATADIR/repo_splits_full
mkdir -p $DSDIR
source $CONDA/etc/profile.d/conda.sh
conda activate $CONDA_ENV
cd src
srun -n 1 python -m DataSet.SplitDataSet --dataset codeparrot/codeparrot-clean \
					 --ratios 0.08 0.01 0.01 0.9 \
					 --splitnames $DSDIR/train $DSDIR/valid $DSDIR/testing $DSDIR/remainder \
					 --strategy "repository" \
					 --repo_field "repo_name" \
					 --loglevel "debug"

## Uncomment the following to immediately continue with pretokenization
## This works as long as DSDIR was empty at the start
## Note that we do not run these in parallel (eventhough we could by appending &)
## because they are at least partially I/O bound anyway.
# echo "Pretokenizing"
# mkdir -p $DATADIR/tokenized
# for split in $(ls $DSDIR)
# do
# 	srun -n 1 python -m DataSet.Pretokenize -t "Salesforce/codegen-350M-multi" -d $DSDIR/$split -o $DATADIR/tokenized/${split}_tokenized.bin -k "content"
# done
