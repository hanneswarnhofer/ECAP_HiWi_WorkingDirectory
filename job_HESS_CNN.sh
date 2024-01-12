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



srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Run.py -e 250 -ft 'latefc' -base 'resnet' -plt 'no'
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Run.py -e 250 -ft 'latemax' -base 'resnet' -plt 'no'

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Run.py -e 250 -ft 'latefc' -base 'moda' -plt 'no'
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Run.py -e 250 -ft 'latemax' -base 'moda' -plt 'no'

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Run.py -e 250 -base 'modamulti' -plt 'no' -single 'no'



#srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Run.py -ft 'latefc' 
#srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Run.py -ft 'latemax'

#srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Run.py -ft 'latefc' -transfer 'yes'
#srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Run.py -ft 'latemax' -transfer 'yes'

#srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Run.py -ft 'latefc' -base 'resnet' -fil 32
#srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Run.py -ft 'latemax' -base 'resnet' -fil 32
#srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Run.py -ft 'latefc' -transfer 'yes' -base 'resnet' -fil 32
#srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Run.py -ft 'latemax' -transfer 'yes' -base 'resnet' -fil 32

#srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Run.py -ft 'latefc' -base 'resnet' -fil 128
#srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Run.py -ft 'latemax' -base 'resnet' -fil 128
#srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Run.py -ft 'latefc' -transfer 'yes' -base 'resnet' -fil 128
#srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Run.py -ft 'latemax' -transfer 'yes' -base 'resnet' -fil 128