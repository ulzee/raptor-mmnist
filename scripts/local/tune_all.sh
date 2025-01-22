#!/bin/bash

# for dset in {vessel}
# for dset in vessel
for dset in {adrenal,fracture,nodule,organ,synapse,vessel}
do
    for lr in {0.00001,0.0001,0.001,0.01,0.1}
    do
        python fit.py --task ${dset}mnist3d_64 --emb-type dec3_${dset}/proj_normal_k100 \
            --model logr --penalty l2 --alpha $lr --save_all_splits
    done
done
