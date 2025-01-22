#!/bin/bash

today=dec8

model=$1
dset=$2

manifest=saved/manifests/${dset}mnist3d.txt

lns=$(wc -l $manifest | awk ' { print $1 }')

bsize=$((lns / 4))

echo $lns $bsize

echo "" > ./scripts/local/by_dset/${dset}.sh

ci=0
for (( i=0; i<=lns; i+=bsize ))
do

    echo "./scripts/local/job.sh /data2/medmnist/${dset}mnist3d_64.npz $model $manifest $i $bsize \
        /data2/bsplat/data/embs/${today}_${model}_${dset} --all_slices --device cuda:$ci" >> \
        ./scripts/local/by_dset/${dset}.sh

    ci=$((ci + 1))
    ci=$((ci % 4))
done
