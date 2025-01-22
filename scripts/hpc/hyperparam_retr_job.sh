#!/bin/bash

. /u/local/Modules/default/init/modules.sh
module load anaconda3
conda activate /u/project/sgss/UKBB/envs/medsam

task=$1
use_emb=$2
layers=$3
temp=$4

echo $task
echo $use_emb
echo $layers
echo $temp

task=${task}mnist3d

python -u fit.py --task $task --emb-type $use_emb \
    --model retr --layers $layers --temperature $temp --save_all_splits --epochs 100 $5 $6 $7 $8

echo done
