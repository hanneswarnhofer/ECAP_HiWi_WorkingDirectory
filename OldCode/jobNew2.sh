#!/bin/bash -l
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

module load python/tensorflow-2.7.0py3.9
pip install --user tables
pip install --user pandas
pip install --user pickle
pip install --user glob
pip install --user sys
pip install --user argparse



srun python /home/hpc/b129dc/b129dc26/MoDA_Project/CTA_Multiview_Analysis_Hannes_Cluster_New.py -n "345" -e 50 -b 512 -r 0.25 -reg 0.001
srun python /home/hpc/b129dc/b129dc26/MoDA_Project/CTA_Multiview_Analysis_Hannes_Cluster_New.py -n "3456" -e 50 -b 512 -r 0.25 -reg 0.001
srun python /home/hpc/b129dc/b129dc26/MoDA_Project/CTA_Multiview_Analysis_Hannes_Cluster_New.py -n "34567" -e 50 -b 512 -r 0.25 -reg 0.001

srun python /home/hpc/b129dc/b129dc26/MoDA_Project/CTA_Multiview_Analysis_Hannes_Cluster_New.py -n "345" -e 50 -b 512 -r 0.25 -reg 0.002
srun python /home/hpc/b129dc/b129dc26/MoDA_Project/CTA_Multiview_Analysis_Hannes_Cluster_New.py -n "3456" -e 50 -b 512 -r 0.25 -reg 0.002
srun python /home/hpc/b129dc/b129dc26/MoDA_Project/CTA_Multiview_Analysis_Hannes_Cluster_New.py -n "34567" -e 50 -b 512 -r 0.25 -reg 0.002

srun python /home/hpc/b129dc/b129dc26/MoDA_Project/CTA_Multiview_Analysis_Hannes_Cluster_New.py -n "345" -e 50 -b 512 -r 0.25 -reg 0.0005
srun python /home/hpc/b129dc/b129dc26/MoDA_Project/CTA_Multiview_Analysis_Hannes_Cluster_New.py -n "3456" -e 50 -b 512 -r 0.25 -reg 0.0005
srun python /home/hpc/b129dc/b129dc26/MoDA_Project/CTA_Multiview_Analysis_Hannes_Cluster_New.py -n "34567" -e 50 -b 512 -r 0.25 -reg 0.0005


