#!/bin/sh
#!/bin/bash
#
#SBATCH --job-name=horse_gelu_lr1e4_hlr1e-5_dn0_dh5e1_svd3_notanh
#SBATCH --output=horse_gelu_lr1e4_hlr1e-5_dn0_dh5e1_svd3_notanh.out
#SBATCH -e horse_gelu_lr1e4_hlr1e-5_dn0_dh5e1_svd3_notanh.err 
#SBATCH -p gypsum-titanx
#SBATCH --mem=60G
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

python -m main --save_file_path 'horse_gelu_lr1e4_hlr1e-5_dn0_dh5e1_svd3_notanh' --lr 1e-5 --arch 'gelu' --norm_delta 0 --hess_delta 5e1 --dropout 0 --losstype 'svd3' --reg 1  --subsample 4096 --grid_N 128 --fname 'horse' --resume_checkpoint 'horse_gelu_lr1e4_dn0_notanh/best_train_loss.tar' --trainfilepath 'data/input/250k_sampled'

sleep 1
exit
