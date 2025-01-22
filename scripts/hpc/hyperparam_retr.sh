#!/bin/bash

# embname=dec14_DINO_DATASET/proj_normal_d1024_k100_run1
# embname=dec26_DINO_DATASET/proj_normal_d1024_k100_run1

mode=${1:-eval}
embname=${2:-dec14_DINO_DATASET/proj_normal_d1024_k100_run1}
datasets=${3:-adrenal,fracture,nodule,organ,synapse,vessel}
gpu=A100
echo $mode
echo $embname
echo $datasets

if [ "$mode" == "tune" ]; then
    for dset in $(echo $datasets | tr ',' '\n')
    do
        # for run in {1,2,3}
        # do
        use_emb=${embname/DATASET/$dset} # some methods won't have multiple "runs"
        # use_emb=${use_emb/run1/"run$run"}

        while IFS= read -r layer
        do
            while IFS= read -r temp
            do
                echo "$dset $layer $temp"
                # Add your processing code here
                qsub -cwd \
                    -N hyp-${layer}-${temp} -l gpu,$gpu,cuda=1,h_rt=01:00:00 -o logs -j y \
                    ./scripts/hpc/hyperparam_retr_job.sh $dset $use_emb $layer $temp $4 $5 $6
                # break
            done < scripts/hpc/temps.txt
            # break
        done < scripts/hpc/layers.txt

        # if [[ $embname == *"VoCo"* || $embname == *"Merlin"* ]]; then
        #     break
        # fi
        # done
    done
fi
# else
#     for dset in $(echo $datasets | tr ',' '\n')
#     do
#         embname=${embname/"/"/"-"}
#         for run in {1,2,3}
#         do
#             use_emb=${embname/run1/"run$run"}
#             python save_best_l2_crossval.py $dset $use_emb ""

#             if [[ $embname == *"VoCo"* || $embname == *"Merlin"* ]]; then
#                 break
#             fi
#         done
#     done
# fi
