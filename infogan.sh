#!/bin/bash
#SBATCH --account=rrg-khisti
#SBATCH --mem=5000M
#SBATCH --account=rrg-khisti
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=1


module load python/3.10
cd ~/project/max-MI-GAN
source venv/bin/activate
python infogan.py