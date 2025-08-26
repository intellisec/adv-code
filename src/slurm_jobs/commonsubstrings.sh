#!/bin/bash
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=16
#SBATCH --mem=92G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gpus-per-node=0        # number of gpus per node
#SBATCH --time=1-00:00:00
#SBATCH --job-name=substrings
#SBATCH --partition=single
#SBATCH --error="unread_logs/slurm-%j_%x.out"
#SBATCH --output="unread_logs/slurm-%j_%x.out"

CONDA=$(conda info --base)
CONDA_ENV=preprocess

DATASET_NAME="codeparrot/codeparrot-clean"
WS_ROOT=$(ws_find train)

OUTDIR=$WS_ROOT/extraction
INFILE=$OUTDIR/stringscomments_50lines.txt
OUTFILE=$OUTDIR/substrings_50lines.txt

mkdir -p $OUTDIR

source $CONDA/etc/profile.d/conda.sh
conda activate $CONDA_ENV
cd src

srun -n 1 python -m Poisoning.CommonSubstrings -o $OUTFILE \
					       -k 25000 \
					       -m 15 \
					       -w \
					       --file $INFILE
