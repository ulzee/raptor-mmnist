#!/bin/bash

for dset in {adrenal,fracture,vessel,synapse,organ,nodule}
do
    cp -r /u/scratch/u/ulzee/raptor/data/embs/dec14_DINO_$dset/proj_normal_d1024_* ../data/embs/dec14_DINO_$dset
done
