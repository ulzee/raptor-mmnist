#!/bin/bash

# ./scripts/abl_embed_mm_sims.sh DINO
# mmroot=/u/scratch/u/ulzee/medmnist
encoder=DINO
GPU=RTX2080Ti
# GPU=A100

EMB_SCRIPT="sam_embed.py"

echo "Launching script: $EMB_SCRIPT"

dset=nodule
# for ssize in {64,32,16,8}
for ssize in 16
do
    saveroot=/u/project/sgss/UKBB/raptor/data/embs/dec22_simulated_location_l$ssize
    task=/u/project/sgss/UKBB/medmnist/simulated/location/l$ssize

    saveto=${saveroot}_${encoder}_${dset}
    mkdir $saveto

    # manifest=saved/manifests/${dset}mnist3d.txt
    manifest=saved/manifests/missing_simulated_shape_l$ssize.txt

    lns=$(wc -l $manifest | awk ' { print $1 }')
    lns=$((lns + 1)) # handle some edge cases with total count

    bsize=$((lns / 3))

    echo $lns $bsize

    for (( i=0; i<=lns; i+=bsize ))
    do
        qsub -cwd -N $GPU-loc-$ssize -l $GPU,gpu,cuda=1,h_rt=04:00:00 -o logs -j y \
            ./scripts/job_embed.sh $EMB_SCRIPT $task $encoder $manifest $i $bsize $saveto "--k 100" --planar $1 $2 $3
        # break
    done
    # break
done
