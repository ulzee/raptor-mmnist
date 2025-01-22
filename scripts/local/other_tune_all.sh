#!/bin/bash

embname=dec16_VoCo_DATASET/avg_cat

for dset in {adrenal,fracture,nodule,organ,synapse,vessel}
do
    use_emb=${embname/DATASET/$dset} # some methods won't have multiple "runs"
    # use_emb=${use_emb/RUN/"run$run"}
    # use_emb=${use_emb/K/"k$K"}

    # for lr in {0.00001,0.0001,0.001,0.01,0.1}
    for lr in {0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0}
    do
        python fit.py --task ${dset}mnist3d --emb-type $use_emb \
            --model logr --penalty l2 --alpha $lr --save_all_splits $1 $2 $3
    done
done
