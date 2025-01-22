#!/bin/bash

embname=dec14_DINO_DATASET/proj_normal_d1024_k100_run1

# for planes in {"A,C","A,S","C,S",A,C,S}
# do
#     # for dset in adrenal
#     for dset in {adrenal,fracture,nodule,organ,synapse,vessel}
#     do
#         use_emb=${embname/DATASET/$dset} # some methods won't have multiple "runs"

#         qsub -cwd -t 1-8:1 \
#             -N tune-${dset} -l h_data=16G,h_rt=00:30:00 -pe shared 8 -o logs -j y \
#             ./scripts/hpc/tune_job.sh $dset $use_emb "--planes $planes"
#         # break
#     done
#     # break
# done


embname=${embname/"/"/"-"}

# after everything finishess...
for planes in {AC,AS,CS,A,C,S}
do
    python save_best_l2_crossval.py $embname -pln$planes
done
