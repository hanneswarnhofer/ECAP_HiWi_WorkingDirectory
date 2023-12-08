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

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 32 -r 0.0001 -reg 0.00005 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 32 -r 0.001 -reg 0.00005 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 32 -r 0.01 -reg 0.00005 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 32 -r 0.1 -reg 0.00005 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 32 -r 0.2 -reg 0.00005 -t 60 -c 2 -ne 100000

'''
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.1 -reg 0.000025 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.125 -reg 0.000025 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.15 -reg 0.000025 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.175 -reg 0.000025 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.2 -reg 0.000025 -t 60 -c 2 -ne 100000

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.1 -reg 0.00001 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.125 -reg 0.00001 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.15 -reg 0.00001 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.175 -reg 0.00001 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.2 -reg 0.00001 -t 60 -c 2 -ne 100000


srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.1 -reg 0.0001 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.125 -reg 0.0001 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.15 -reg 0.0001 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.175 -reg 0.0001 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.2 -reg 0.0001 -t 60 -c 2 -ne 100000

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.15 -reg 0.00005 -t 0 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.15 -reg 0.00005 -t 20 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.15 -reg 0.00005 -t 40 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.15 -reg 0.00005 -t 80 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.15 -reg 0.00005 -t 100 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.15 -reg 0.00005 -t 120 -c 2 -ne 100000

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.15 -reg 0.00005 -t 0 -c 3 -ne 10000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.15 -reg 0.00005 -t 20 -c 3 -ne 10000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.15 -reg 0.00005 -t 40 -c 3 -ne 10000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.15 -reg 0.00005 -t 80 -c 3 -ne 10000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.15 -reg 0.00005 -t 100 -c 3 -ne 10000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.15 -reg 0.00005 -t 120 -c 3 -ne 10000

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.15 -reg 0.00005 -t 0 -c 4 -ne 10000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.15 -reg 0.00005 -t 20 -c 4 -ne 10000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.15 -reg 0.00005 -t 40 -c 4 -ne 10000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.15 -reg 0.00005 -t 80 -c 4 -ne 10000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.15 -reg 0.00005 -t 100 -c 4 -ne 10000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_new.py -e 50 -b 1024 -r 0.15 -reg 0.00005 -t 120 -c 4 -ne 10000
'''