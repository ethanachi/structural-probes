#!/bin/bash

#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --job-name=run-sp-experiment-{filename}
#SBATCH --mem=16G
#SBATCH --open-mode=append
#SBATCH --partition=jag-standard
#SBATCH --time=10-0
#SBATCH --output=sl/slurm-%j.out

# activate your desired anaconda environment
source activate sp

# cd to working directory
cd .

# launch commands
python ~/structural-probes/structural-probes/run_experiment.py ~/structural-probes/tests/multiling/experiments/{filename}


