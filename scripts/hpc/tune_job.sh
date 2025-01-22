#!/bin/bash

. /u/local/Modules/default/init/modules.sh
module load anaconda3
conda activate /u/project/sgss/UKBB/envs/medsam

task=$1
use_emb=$2

alpha_ix=$SGE_TASK_ID
alpha=$(sed -n "${alpha_ix}p" ./scripts/hpc/alphas.txt)

echo "Alpha: $alpha"

echo $task
echo $use_emb

if [[ $task != *"location"* && $task != *"shape"* ]]; then
    task=${task}mnist3d
    # task=$dset
fi

python -u fit.py --task $task --emb-type $use_emb \
    --model logr --penalty l2 --alpha $alpha --save_all_splits $3 $4 $5 $6

echo done
