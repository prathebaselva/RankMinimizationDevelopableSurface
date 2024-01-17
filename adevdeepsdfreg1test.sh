#!/bin/sh
#!/bin/bash
#
#SBATCH --job-name=abc4303_latmodel_reg1_dh1e1_lr1e4_h5e-5_hessiandethat
#SBATCH --output=abc4303_latmodel_reg1_dh1e1_lr1e4_h5e-5_hessiandethat.out
#SBATCH -e abc4303_latmodel_reg1_dh1e1_lr1e4_h5e-5_hessiandethat.err
#SBATCH -p gypsum-titanx
#SBATCH --mem=50G
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

python -m maindeepsdftest --save_file_path 'output/abc/test/abc4303_latmodel_reg1_dh1e1_lr1e4_hessiandethat' --lr 5e-5 --arch 'gelu' --norm_delta 0 --hess_delta 1e1 --dropout 0 --losstype 'hessiandethat' --reg 1  --subsample 4096 --totalsamples 18022  --grid_N 128 --fname 'abc4303' --resume_checkpoint 'abc4303_latmodel_reg0_dh1e1_lr1e4_hessiandethat/best_train_loss.tar' --testindex 3 --opttype 'latmodel' --resume 1 --epochs 1000

sleep 1
exit
