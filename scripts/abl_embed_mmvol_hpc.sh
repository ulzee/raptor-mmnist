#!/bin/bash


saveroot=/u/scratch/u/ulzee/brainsplat/data/embs/dec17

# for encoder in {Merlin,VoCo}
for encoder in {SAM,MedSAM,CLIP,DINO}
do
    ./scripts/embed_mmvol_hpc.sh $encoder $saveroot --avgpool
    for K in {10,100}
    do
        ./scripts/embed_mmvol_hpc.sh $encoder $saveroot "--k $K" --planar
    done
    # ./scripts/embed_mmvol_hpc.sh $encoder $saveroot --planar # this might take a lot of space
done
