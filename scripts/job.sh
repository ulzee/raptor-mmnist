#!/bin/bash

# . /u/local/Modules/default/init/modules.sh
# module load anaconda3
# conda activate /u/project/sgss/UKBB/envs/medsam2

python -u sam_embed.py --dataroot /u/scratch/u/ulzee/medmnist/octmnist_224.npz \
    --encoder $1 --manifest $2 --start $3 --many $4 --batch_size 32 --saveto $5 $6 $7 $8

echo done
