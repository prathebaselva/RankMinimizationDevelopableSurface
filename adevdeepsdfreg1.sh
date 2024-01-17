#!/bin/sh
#!/bin/bash
#
#SBATCH --job-name=abc_gelu_lr1e4_hlr1e-5_dn0_dh1e3_svd3_notanh
#SBATCH --output=abc_gelu_lr1e4_hlr1e-5_dn0_dh1e3_svd3_notanh.out
#SBATCH -e abc_gelu_lr1e4_hlr1e-5_dn0_dh1e3_svd3_notanh.err 
#SBATCH -p gypsum-2080ti
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

python -m maindeepsdf --save_file_path 'output/abc/abcreg1' --lr 1e-5 --arch 'gelu' --norm_delta 0 --hess_delta 1e3 --dropout 0 --losstype 'svd3' --reg 1  --subsample 4096 --totalsamples 18022  --grid_N 128 --fname 'abc' --resume_checkpoint 'abc/best_train_loss.tar' 

sleep 1
exit
