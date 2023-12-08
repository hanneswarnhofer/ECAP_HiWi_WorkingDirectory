#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --export=NONE


unset SLURM_EXPORT_ENV



module load python/3.9-anaconda
module load cuda/11.8.0
module load cudnn/8.6.0.163-11.8
module load tensorrt/8.5.3.1-cuda11.8-cudnn8.6


micromamba activate dl1dh

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Single_Freeze_Multi.py -e 50 -b 64 -r 0.2 -reg 0.001 -base 'resnet' 
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Single_Freeze_Multi.py -e 50 -b 64 -r 0.2 -reg 0.002 -transfer 'yes' -base 'resnet'
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Single_Freeze_Multi.py -e 50 -b 64 -r 0.2 -reg 0.001
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Single_Freeze_Multi.py -e 50 -b 64 -r 0.2 -reg 0.002 -transfer 'yes'
'''
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Single_Freeze_Multi.py -e 50 -b 64 -r 0.2 -reg 0.00005 -transfer 'yes'
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Single_Freeze_Multi.py -e 50 -b 64 -r 0.2 -reg 0.0001 -transfer 'yes'
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Single_Freeze_Multi.py -e 50 -b 64 -r 0.2 -reg 0.00025 -transfer 'yes'
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Single_Freeze_Multi.py -e 50 -b 64 -r 0.2 -reg 0.0005 -transfer 'yes'
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Single_Freeze_Multi.py -e 50 -b 64 -r 0.2 -reg 0.001 -transfer 'yes'
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Single_Freeze_Multi.py -e 50 -b 64 -r 0.2 -reg 0.002 -transfer 'yes'

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Single_Freeze_Multi.py -e 50 -b 64 -r 0.2 -reg 0.00005
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Single_Freeze_Multi.py -e 50 -b 64 -r 0.2 -reg 0.0001
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Single_Freeze_Multi.py -e 50 -b 64 -r 0.2 -reg 0.00025
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Single_Freeze_Multi.py -e 50 -b 64 -r 0.2 -reg 0.0005
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Single_Freeze_Multi.py -e 50 -b 64 -r 0.2 -reg 0.001
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Single_Freeze_Multi.py -e 50 -b 64 -r 0.2 -reg 0.002

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Single_Freeze_Multi.py -e 50 -b 64 -r 0.02 -reg 0.00005 -transfer 'yes'
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Single_Freeze_Multi.py -e 50 -b 64 -r 0.02 -reg 0.0001 -transfer 'yes'
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Single_Freeze_Multi.py -e 50 -b 64 -r 0.02 -reg 0.00025 -transfer 'yes'
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Single_Freeze_Multi.py -e 50 -b 64 -r 0.02 -reg 0.0005 -transfer 'yes'
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Single_Freeze_Multi.py -e 50 -b 64 -r 0.02 -reg 0.001 -transfer 'yes'
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Single_Freeze_Multi.py -e 50 -b 64 -r 0.02 -reg 0.002 -transfer 'yes'

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Single_Freeze_Multi.py -e 50 -b 64 -r 0.02 -reg 0.00005
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Single_Freeze_Multi.py -e 50 -b 64 -r 0.02 -reg 0.0001
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Single_Freeze_Multi.py -e 50 -b 64 -r 0.02 -reg 0.00025
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Single_Freeze_Multi.py -e 50 -b 64 -r 0.02 -reg 0.0005
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Single_Freeze_Multi.py -e 50 -b 64 -r 0.02 -reg 0.001
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Single_Freeze_Multi.py -e 50 -b 64 -r 0.02 -reg 0.002
'''