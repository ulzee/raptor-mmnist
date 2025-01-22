#!/bin/bash

ukbbroot=/u/project/sgss/UKBB/imaging/bulk
encoder=$1
saveroot=$2
dset=$3 # 20253 (brain), XXX (liver)
GPU=RTX2080Ti
use_manifest=main

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

task=$ukbbroot/$dset
saveto=${saveroot}_${encoder}_${dset}
mkdir $saveto


manifest=saved/manifests/ukbb-wbu-$dset-filelist.txt

lns=$(wc -l $manifest | awk ' { print $1 }')
if [ $lns -eq 0 ]; then
    continue
fi
lns=$((lns + 1)) # handle some edge cases with total count

bsize=$((lns / 8))

echo $lns $bsize
if [ $bsize -eq 0 ]; then
    bsize=1
fi

for (( i=0; i<=lns; i+=bsize ))
do
    qsub -cwd -N ${GPU}-${encoder}-${dset} -l $GPU,gpu,cuda=1,h_rt=05:00:00 -o logs -j y \
        ./scripts/job_embed.sh $EMB_SCRIPT $task $encoder $manifest $i $bsize $saveto $4 $5 $6 $7 $8
    break
done