#!/bin/bash

datasets=${1:-adrenal,fracture,nodule,organ,synapse,vessel}
etc=$2
gpu=$3

echo $datasets

for dset in $(echo $datasets | tr ',' '\n')
do
    qsub -cwd \
        -N mae-${dset} -l gpu,cuda=1,$gpu,h_rt=01:30:00 -o logs -j y \
        ./scripts/hpc/fit_mae_job.sh --task ${dset}mnist3d --batch-size 128 $etc
    # break
done
