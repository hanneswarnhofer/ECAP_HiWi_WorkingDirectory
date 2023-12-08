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


srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_CustomResNet_simpler.py -ft 'latefc' 
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_CustomResNet_simpler.py -ft 'latemax'
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_CustomResNet_simpler.py -ft 'latefc' -transfer 'yes'
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_CustomResNet_simpler.py -ft 'latemax' -transfer 'yes'

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_CustomResNet_simpler.py -ft 'earlymaxB0' 
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_CustomResNet_simpler.py -ft 'earlyconvB0'
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_CustomResNet_simpler.py -ft 'earlymaxB0' -transfer 'yes'
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_CustomResNet_simpler.py -ft 'earlyconvB0' -transfer 'yes'

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_CustomResNet_simpler.py -ft 'latefc' -lr 0.00001 -reg 0.001
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_CustomResNet_simpler.py -ft 'latemax' -lr 0.00001 -reg 0.001
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_CustomResNet_simpler.py -ft 'latefc' -transfer 'yes' -lr 0.00001 -reg 0.001
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_CustomResNet_simpler.py -ft 'latemax' -transfer 'yes' -lr 0.00001 -reg 0.001

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_CustomResNet_simpler.py -ft 'earlymaxB0' -lr 0.00001 -reg 0.001
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_CustomResNet_simpler.py -ft 'earlyconvB0' -lr 0.00001 -reg 0.001
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_CustomResNet_simpler.py -ft 'earlymaxB0' -transfer 'yes' -lr 0.00001 -reg 0.001
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_CustomResNet_simpler.py -ft 'earlyconvB0' -transfer 'yes' -lr 0.00001 -reg 0.001

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_CustomResNet_simpler.py -ft 'latefc' -lr 0.001
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_CustomResNet_simpler.py -ft 'latemax' -lr 0.001
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_CustomResNet_simpler.py -ft 'latefc' -transfer 'yes' -lr 0.001
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_CustomResNet_simpler.py -ft 'latemax' -transfer 'yes' -lr 0.001

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_CustomResNet_simpler.py -ft 'earlymaxB0' -lr 0.001
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_CustomResNet_simpler.py -ft 'earlyconvB0' -lr 0.001
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_CustomResNet_simpler.py -ft 'earlymaxB0' -transfer 'yes' -lr 0.001
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_CustomResNet_simpler.py -ft 'earlyconvB0' -transfer 'yes' -lr 0.001
