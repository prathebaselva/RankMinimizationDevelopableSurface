#!/bin/sh
#!/bin/bash
#
#SBATCH --job-name=lucy_gelu_lr1e4_hlr1e-5_dn0_svd3_notanh_resume
#SBATCH --output=lucy_gelu_lr1e4_hlr1e-5_dn0_notanh_svd3_resume.out
#SBATCH -e lucy_gelu_lr1e4_hlr1e-5_dn0_notanh_svd3_resume.err 
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

python -m main --save_file_path 'lucy_gelu_lr1e4_hlr1e-5_dn0_dh1e1_svd3_notanh' --lr 1e-5 --arch 'gelu' --norm_delta 0 --subsample 4096 --fname 'lucy' --grid_N 128 --resume_checkpoint 'lucy_gelu_lr1e4_hlr1e-5_dn0_dh1e1_svd3_notanh/best_train_loss.tar' --resume True --reg 1



sleep 1
exit
