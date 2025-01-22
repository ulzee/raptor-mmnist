#!/bin/bash

encoder=DINO
saveroot=/u/project/sgss/UKBB/raptor/data/embs/dec14
# saveroot=/u/scratch/u/ulzee/raptor/data/embs/dec14

datasets=$1

# ./scripts/embed_mmvol_hpc.sh $encoder $saveroot $datasets "--k 1,5,50" --planar
./scripts/embed_mmvol_hpc.sh $encoder $saveroot $datasets "--k 10,100,150" --planar
