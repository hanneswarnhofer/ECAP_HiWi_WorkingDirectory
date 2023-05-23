#!/bin/bash -l
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
module load python/tensorflow-2.7.0py3.9
pip install --user optuna
pip install --user tables

srun python /home/hpc/b129dc/b129dc17/scripts/fixedArch/23_2/Stereo_fixed_23_correctedVal.py 