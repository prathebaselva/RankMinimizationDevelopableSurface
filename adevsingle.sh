#!/bin/sh
#!/bin/bash
#
#SBATCH --job-name=adev_tanh_lr1e4_clamp01_drop11_pat10
#SBATCH --output=adev_tanh_lr1e4_clamp01_drop11_pat10.out
#SBATCH -e adev_tanh_lr1e4_clamp01_drop11_pat10.err 
#SBATCH -p gypsum-titanx
#SBATCH --mem=20G
#SBATCH --exclude=gypsum-gpu043

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=7-00:00         # Maximum runtime in D-HH:MM
#SBATCH --gres=gpu:1
#SBATCH --mail-user pselvaraju@cs.umass.edu

#module load gcc/7.1.0
#module load cuda11/11.2.1
#module load cudnn/7.5-cuda_999.2
module load cuda11/11.2.1

python -m main --save_file_path 'tanh_lr1e4_clamp01_drop11_pat10' --lr 1e-4 --latlr 1e-5 --arch 'tanh' --patience 10 --dropout 1 --lat_delta 1e-4


sleep 1
exit
