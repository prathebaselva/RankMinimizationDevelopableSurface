#!/bin/sh
#!/bin/bash
#
#SBATCH --job-name=bunny_high_gelu_lr1e4_dn0_notanh_grid1024_resume
#SBATCH --output=bunny_high_gelu_lr1e4_dn0_notanh_grid1024_resume.out
#SBATCH -e bunny_high_gelu_lr1e4_dn0_notanh_grid1024_resume.err 
#SBATCH -p gypsum-1080ti
#SBATCH --mem=100G
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

python -m main --save_file_path 'bunny_high_gelu_lr1e4_dn0_notanh_resume' --lr 1e-5 --arch 'gelu' --norm_delta 0 --subsample 4096 --fname 'bunny_high' --grid_N 1024 --resume_checkpoint 'bunny_high_gelu_lr1e4_dn0_notanh/best_train_loss.tar' --resume True --reg 1



sleep 1
exit
