#!/bin/bash

# embname=dec14_DINO_DATASET/proj_normal_d1024_k100_run1
# embname=dec26_DINO_DATASET/proj_normal_d1024_k100_run1

mode=${1:-eval}
embname=${2:-dec14_DINO_DATASET/proj_normal_d1024_k100_run1}
datasets=${3:-adrenal,fracture,nodule,organ,synapse,vessel}

echo $mode
echo $embname
echo $datasets

if [ "$mode" == "tune" ]; then
    for dset in $(echo $datasets | tr ',' '\n')
    do
        for run in {1,2,3}
        do
            use_emb=${embname/DATASET/$dset} # some methods won't have multiple "runs"
            use_emb=${use_emb/run1/"run$run"}

            qsub -cwd -t 1-10:1 \
                -N tune-${dset} -l h_data=8G,h_rt=00:30:00 -pe shared 4 -o logs -j y \
                ./scripts/hpc/tune_job.sh $dset $use_emb $4 $5 $6
        # break
            if [[ $embname == *"VoCo"* || $embname == *"Merlin"* ]]; then
                break
            fi
        done
    done
else
    for dset in $(echo $datasets | tr ',' '\n')
    do
        embname=${embname/"/"/"-"}
        for run in {1,2,3}
        do
            use_emb=${embname/run1/"run$run"}
            python save_best_l2_crossval.py $dset $use_emb $4 $5 $6

            if [[ $embname == *"VoCo"* || $embname == *"Merlin"* ]]; then
                break
            fi
        done
    done
fi
