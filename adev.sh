#!/bin/sh
#!/bin/bash
#
#SBATCH --job-name=horse_oded_gelu_lr1e4_dn0_notanh
#SBATCH --output=horse_oded_gelu_lr1e4_dn0_notanh.out
#SBATCH -e horse_oded_gelu_lr1e4_dn0_notanh.err 
#SBATCH -p gypsum-titanx
#SBATCH --mem=250G
#SBATCH --exclude=gypsum-gpu043,gypsum-gpu085,gypsum-gpu086,gypsum-gpu087

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=7-00:00         # Maximum runtime in D-HH:MM
#SBATCH --gres=gpu:1
#SBATCH --mail-user pselvaraju@cs.umass.edu

#module load gcc/7.1.0
#module load cuda11/11.2.1
#module load cudnn/7.5-cuda_999.2
module load cuda11/11.2.1

python -m main --save_file_path 'output/horse_oded_gelu_lr1e4_dn0_notanh' --lr 1e-4 --arch 'gelu' --norm_delta 0 --subsample 16384 --fname 'horse_oded' --grid_N 128 --omega 1 --totalsamples 275000 --trainfilepath 'data/input/250k_sampled' --dropout 0


sleep 1
exit
