#!/bin/bash

mmroot=/u/project/sgss/UKBB/medmnist
encoder=$1
saveroot=$2
datasets=$3
GPU=RTX2080Ti
# GPU=H100
# GPU=A100
use_manifest=main
# use_manifest=missing

EMB_SCRIPT="embed.py"
# if [ "$encoder" == "VoCo" ] || [ "$encoder" == "Merlin" ]; then
if [ "$encoder" == "VoCo" ]; then
    EMB_SCRIPT="vol_embed.py"
    # GPU=A100
fi
if [ "$encoder" == "Merlin" ]; then
    EMB_SCRIPT="vol_embed.py"
fi

echo "Launching script: $EMB_SCRIPT"

# for dset in nodule
# for dset in {organ,fracture,vessel,synapse}
# for dset in {adrenal,synapse,organ}
# for dset in nodule

for dset in $(echo $datasets | tr ',' '\n')
do
    echo $dset
    if [[ $saveroot != *"simulated"* ]]; then
        task=$mmroot/${dset}mnist3d_64.npz
    else
        simtype=$(echo $saveroot | awk -F_ '{print $NF}')
        task=$mmroot/simulated/$simtype
    fi
    saveto=${saveroot}_${encoder}_${dset}
    mkdir $saveto

    if [ "$use_manifest" == "missing" ]; then
        manifest=saved/manifests/missing_${encoder}_${dset}.txt
    else
        manifest=saved/manifests/${dset}mnist3d.txt
    fi


    lns=$(wc -l $manifest | awk ' { print $1 }')
    if [ $lns -eq 0 ]; then
        continue
    fi
    lns=$((lns + 1)) # handle some edge cases with total count

    # qsub -cwd -N emb-${dset} -l $GPU,gpu,cuda=1,h_rt=05:00:00 -o logs -j y \
    #     ./scripts/job_embed.sh $EMB_SCRIPT $task $encoder $manifest 0 $lns $saveto $4 $5 $6 $7 $8

    # break
    # if [ "$encoder" == "VoCo" ] || [ "$encoder" == "Merlin" ]; then
    #     # since vol embedders are fast, launch one job per
    # fi

    # bsize=$((lns / 4))
    bsize=$((lns / 8))
    # bsize=$lns

    echo $lns $bsize
    if [ $bsize -eq 0 ]; then
        bsize=1
    fi

    # for (( i=bsize*6; i<=lns; i+=bsize ))
    for (( i=0; i<=lns; i+=bsize ))
    do
        qsub -cwd -N ${GPU}-${encoder}-${dset} -l $GPU,gpu,cuda=1,h_rt=05:00:00 -o logs -j y \
            ./scripts/job_embed.sh $EMB_SCRIPT $task $encoder $manifest $i $bsize $saveto $4 $5 $6 $7 $8
        # qsub -cwd -N emb-${dset} -l $GPU,gpu,cuda=1,h_rt=03:00:00 -o logs -j y \
        #     ./scripts/job_embed.sh $EMB_SCRIPT $mmroot/${dset}mnist3d_64.npz $encoder $manifest $i $bsize $saveto --all_slices $4 $5 $6 $7 $8
        # break
    done
    # # break
done
