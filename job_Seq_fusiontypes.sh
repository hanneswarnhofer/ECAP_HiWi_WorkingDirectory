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

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -ft "latefc" 
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -ft "latemax" 
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -ft "earlymax" 
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -ft "earlyconv" 

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -ft "latefc" -n "norm"
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -ft "latemax" -n "norm"
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -ft "earlymax" -n "norm"
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -ft "earlyconv" -n "norm"

'''

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 8 -r 0.00001 -reg 0.00005 -t 60 -c 2 -ne 100000 
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 8 -r 0.0001 -reg 0.00005 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 8 -r 0.001 -reg 0.00005 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 8 -r 0.01 -reg 0.00005 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 8 -r 0.1 -reg 0.00005 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 8 -r 0.2 -reg 0.00005 -t 60 -c 2 -ne 100000

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 32 -r 0.00001 -reg 0.00005 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 32 -r 0.0001 -reg 0.00005 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 32 -r 0.001 -reg 0.00005 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 32 -r 0.01 -reg 0.00005 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 32 -r 0.1 -reg 0.00005 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 32 -r 0.2 -reg 0.00005 -t 60 -c 2 -ne 100000

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 64 -r 0.00001 -reg 0.00005 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 64 -r 0.0001 -reg 0.00005 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 64 -r 0.001 -reg 0.00005 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 64 -r 0.01 -reg 0.00005 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 64 -r 0.1 -reg 0.00005 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 64 -r 0.2 -reg 0.00005 -t 60 -c 2 -ne 100000

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 1024 -r 0.1 -reg 0.00001 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 1024 -r 0.125 -reg 0.00001 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 1024 -r 0.15 -reg 0.00001 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 1024 -r 0.175 -reg 0.00001 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 1024 -r 0.2 -reg 0.00001 -t 60 -c 2 -ne 100000

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 32 -r 0.1 -reg 0.00001 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 32 -r 0.125 -reg 0.00001 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 32 -r 0.15 -reg 0.00001 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 32 -r 0.175 -reg 0.00001 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 32 -r 0.2 -reg 0.00001 -t 60 -c 2 -ne 100000

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 1024 -r 0.1 -reg 0.000025 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 1024 -r 0.125 -reg 0.000025 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 1024 -r 0.15 -reg 0.000025 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 1024 -r 0.175 -reg 0.000025 -t 60 -c 2 -ne 100000
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Alex_Seq.py -e 50 -b 1024 -r 0.2 -reg 0.000025 -t 60 -c 2 -ne 100000

'''
