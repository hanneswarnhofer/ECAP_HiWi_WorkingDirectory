#!/bin/bash -l
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --export=NONE


unset SLURM_EXPORT_ENV

module load python/3.9-anaconda

source activate myenv

pip install --user tables
pip install --user pandas
pip install --user pickle
pip install --user glob
pip install --user sys
pip install --user argparse
pip install --user ctapipe
pip install --user cta-extra
pip install --user dl1_data_handler
pip install --user tensorflow

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex.py 

