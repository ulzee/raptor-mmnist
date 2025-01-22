#!/bin/bash

# saveroot=/u/project/sgss/UKBB/raptor/data/embs/dec26
saveroot=/u/project/sgss/UKBB/raptor/data/embs/dec24
datasets=nodule,fracture,vessel,synapse,organ,adrenal
# datasets=vessel

# for encoder in SAM
# for encoder in CLIP
# for encoder in DINO
for encoder in {SAM,MedSAM,CLIP}
do
    ./scripts/embed_mmvol_hpc.sh $encoder $saveroot $datasets "--k 100" --planar
    # break
done
