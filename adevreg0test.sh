#!/bin/sh
#!/bin/bash
#
#SBATCH --job-name=griffin_gaussthin_2_25_gelu_lr1e4_dn0_notanh_test
#SBATCH --output=griffin_gaussthin_2_25_gelu_lr1e4_dn0_notanh_test.out
#SBATCH -e griffin_gaussthin_2_25_gelu_lr1e4_dn0_notanh_test.err 
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

python -m main --save_file_path 'npyfiles/griffin_gaussthin_2_25_gelu_lr1e4_dn0_notanh_test' --lr 1e-5 --arch 'gelu' --norm_delta 0 --hess_delta 3 --dropout 0 --losstype 'hessiandethat' --reg 1  --subsample 4096 --grid_N 128 --fname 'griffin_gaussthin_2_25' --resume_checkpoint 'griffin_gaussthin_2_25_gelu_lr1e4_dn0_notanh/best_train_loss.tar' --trainfilepath 'data/input/500k_sampled' --test True 



sleep 1
exit
