#!/bin/bash

datasets=${1:-adrenal,fracture,nodule,organ,synapse,vessel}
etc=$2

for dset in $(echo $datasets | tr ',' '\n')
do
    # python fit_e2e.py --task ${dset}mnist3d --model suprem --epochs 20 --lr 1e-5 $etc
    qsub -cwd \
        -N suprem-${dset} -l gpu,cuda=1,A100,h_rt=03:00:00 -o logs -j y \
        ./scripts/hpc/fit_e2e_job.sh --task ${dset}mnist3d --model suprem --lr 1e-5 --epochs 20 $etc
done