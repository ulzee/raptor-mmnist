#!/bin/bash

encoder=SAM
bsize=300
# bsize=10000

saveto=$1

lns=$(wc -l artifacts/20253_more.txt | awk ' { print $1 }')
echo $lns

for (( i=0; i<=lns; i+=bsize ))
do
    echo $i
    qsub -cwd -N emb -l RTX2080Ti,gpu,cuda=1,h_rt=02:00:00 -o logs -j y \
        ./scripts/job_more.sh $encoder $i $bsize $saveto $2 $3 $4
done
