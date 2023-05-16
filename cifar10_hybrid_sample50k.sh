#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu_quad
#SBATCH -t 7:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --array=1-8
#SBATCH --mail-user=binxu_wang@hms.harvard.edu
#SBATCH -o cifar_hyrbid_samp_%j.out

echo "$SLURM_ARRAY_TASK_ID"

param_list=\
'--seed_start 11000 --seed_end 16000
--seed_start 16000 --seed_end 21000
--seed_start 21000 --seed_end 26000
--seed_start 26000 --seed_end 31000
--seed_start 31000 --seed_end 36000
--seed_start 36000 --seed_end 41000
--seed_start 41000 --seed_end 46000
--seed_start 46000 --seed_end 51000
'

export unit_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$unit_name"

module load gcc/6.2.0
module load cuda/10.2
#module load conda2/4.2.13

#conda init bash
source  activate torch

cd ~/Github/DiffusionSpectra
python3 core/CIFAR10_PCA_analytical_DDIM_hybrid_O2.py  $unit_name
