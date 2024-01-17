#!/bin/sh
#!/bin/bash
#
#SBATCH --job-name=deepsdf
#SBATCH --output=deepsdf.out
#SBATCH -e deepsdf.err 
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

python -m maindeepsdf --save_file_path 'output/abc/abc' --lr 1e-4 --arch 'gelu' --norm_delta 0 --subsample 16384 --fname 'abc' --grid_N 128 --omega 1 --totalsamples 18022 --dropout 0 --resume_checkpoint 'abc/best_train_loss.tar'


sleep 1
exit
