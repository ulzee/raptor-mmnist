#!/bin/bash

. /u/local/Modules/default/init/modules.sh
module load anaconda3

# NOTE: make sure $4 is the model name
if [ "$4" == "pctnet" ] || [ "$4" == "resnet50" ]; then
    conda activate /u/scratch/u/ulzee/envs/misfm
else
    conda activate /u/project/sgss/UKBB/envs/medsam2
fi

nvidia-smi

python -u fit_e2e.py $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15}
