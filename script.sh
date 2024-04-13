#!/bin/bash -l
#SBATCH --job-name=contrimix
#SBATCH --partition=all_serial #all_usr_prod invece è la partizione di produzione, che è dove si eseguono le cose quando sono ok
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2 #o quello che volete, vi consiglio 2
#SBATCH --mem=10G #se non bastano chiedetene di più, questi sono i GB di RAM, non della GPU (che sono fissi)
#SBATCH --time=0-15:00:00 #queste sono 10 ore, cambiate come vi pare, max sono 24 ore
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --account=cmattace #vostro account

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate new

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

conda install numpy wilds scikit-learn scipy seaborn tqdm transformers poetry

export PYTHONPATH=/homes/cmattace/ContriMix/ip_drit
pip install absl-py wandb catalyst
pip install grad-cam

srun python /homes/cmattace/ContriMix/camelyon_17_do_all_contrimix.py --run_on_cluster False --dataset_dir_local '/' --log_dir_local '/homes/cmattace/logs'