#!/bin/bash

# embname=dec16_VoCo_DATASET/avg_cat
# embname=dec16_VoCo_DATASET/planar_avg_cat
# embname=dec16_VoCo_DATASET/raw
embname=dec16_Merlin_DATASET/raw

# for dset in {adrenal,fracture}
for dset in {adrenal,fracture,nodule,organ,synapse,vessel}
do
    use_emb=${embname/DATASET/$dset} # some methods won't have multiple "runs"

    qsub -cwd -t 1-8:1 \
        -N tune-${dset} -l h_data=16G,h_rt=00:30:00 -pe shared 8 -o logs -j y \
        ./scripts/hpc/tune_job.sh $dset $use_emb $1 $2 $3
        # -N tune-${dset} -l h_data=32G,h_rt=00:30:00 -pe shared 32 -o logs -j y \
    # break
done
