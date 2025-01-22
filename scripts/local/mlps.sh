#!/bin/bash

# wdecay=$1
# dropout=$2

for dset in {adrenal,fracture,nodule,organ,synapse,vessel}
do
    python fit.py --task ${dset}mnist3d_64 --emb-type dec3_${dset}/proj_normal_k10 \
        --model mlp $1 $2 $3 $4 $5 $6 #--weight-decay $wdecay --dropout $dropout
done
