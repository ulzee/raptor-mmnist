#!/bin/bash

datasets=${1:-adrenal,fracture,nodule,organ,synapse,vessel}
etc=$2

echo $datasets

for dset in $(echo $datasets | tr ',' '\n')
do
    qsub -cwd \
        -N pctnet-${dset} -l gpu,cuda=1,A100,h_rt=03:00:00 -o logs -j y \
        ./scripts/hpc/fit_e2e_job.sh --task ${dset}mnist3d --model pctnet --batch-size 4 --lr 1e-4 --epochs 10 $etc
done
