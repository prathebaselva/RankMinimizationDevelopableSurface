#!/bin/sh
#!/bin/bash
#
#SBATCH --job-name=test_adev_tanh_lr1e4_latlr1e4_clamp01_drop1_epochsched_pat10
#SBATCH --output=test_adev_tanh_lr1e4_latlr1e4_clamp01_drop1_epochsched_pat10.out
#SBATCH -e test_adev_tanh_lr1e4_latlr1e4_clamp01_drop1_epochsched_pat10.err 
#SBATCH -p gypsum-titanx
#SBATCH --mem=20G
#SBATCH --exclude=gypsum-gpu043
#SBATCH --array=1-1

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=7-00:00         # Maximum runtime in D-HH:MM
#SBATCH --gres=gpu:1
#SBATCH --mail-user pselvaraju@cs.umass.edu

#module load gcc/7.1.0
#module load cuda11/11.2.1
#module load cudnn/7.5-cuda_999.2
module load cuda11/11.2.1

python -m maintest --save_file_path 'tanh_lr1e4_latlr1e4_clamp01_drop1_epochsched_pat10' --lr 1e-4 --latlr 1e-1 --arch 'tanh' --patience 10 --dropout 1 --lat_delta 1e-4 --testfileindex ${SLURM_ARRAY_TASK_ID} --resume_checkpoint 'tanh_lr1e4_latlr1e4_clamp01_drop1_epochsched_pat10/best_train_loss.tar' --epochs 300000 --resume True


sleep 1
exit
