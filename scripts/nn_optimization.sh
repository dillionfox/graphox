#!/bin/bash
#SBATCH --job-name=ray_tune_hp_opt
#SBATCH --cpus-per-task=52
#SBATCH --mem-per-cpu 9000
#SBATCH --gpus 2
#SBATCH --time 72:00:00
#SBATCH --output=hp_opt_%j.log   # Standard output and error log

module load anaconda3
module load cuda11.3

source activate
conda activate graphox

cd /home/dfox/code/graphox

date
echo "....................running...................."
python graphox/rgcn/opt.py > opt.out
echo "...................complete...................."
date