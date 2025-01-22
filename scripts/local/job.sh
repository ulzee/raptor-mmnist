#!/bin/bash

# . /u/local/Modules/default/init/modules.sh
# module load anaconda3
# conda activate /u/project/sgss/UKBB/envs/medsam2

python -u sam_embed.py --dataroot $1 \
    --encoder $2 --manifest $3 --start $4 --many $5 --batch_size 257 \
    --saveto $6 $7 $8 $9 ${10} ${11} ${12}

echo done
