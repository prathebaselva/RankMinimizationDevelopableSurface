#!/bin/sh
#!/bin/bash
#
#SBATCH --job-name=abc4305_lat_reg0_dh1e1_lr1e4_hessiandethat
#SBATCH --output=abc4305_lat_reg0_dh1e1_lr1e4_hessiandethat.out
#SBATCH -e abc4305_lat_reg0_dh1e1_lr1e4_hessiandethat.err
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

python -m maindeepsdftest --save_file_path 'output/abc/test/abc4305_lat_reg0_dh1e1_lr1e4_hessiandethat' --lr 1e-4 --arch 'gelu' --norm_delta 0 --hess_delta 1e1 --dropout 0 --losstype 'hessiandethat' --reg 0  --subsample 4096 --totalsamples 18022  --grid_N 128 --fname 'abc4305' --resume_checkpoint 'abc/best_train_loss.tar' --testindex 5 --opttype 'lat' --epoch 1000

sleep 1
exit
