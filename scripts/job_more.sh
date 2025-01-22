#!/bin/bash

. /u/local/Modules/default/init/modules.sh
module load anaconda3
conda activate /u/project/sgss/UKBB/envs/medsam

python -u medsam_embed.py --encoder $1 --manifest artifacts/20253_more.txt --start $2 --many $3 --batch_size 64 --saveto $4 $5 $6
# python -u medsam_embed.py --encoder $1 --manifest artifacts/20253_more.txt --start $2 --many $3 --batch_size 512 --saveto $4 $5 $6

echo done
