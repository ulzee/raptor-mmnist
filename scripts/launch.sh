#!/bin/bash


manifest=$1

encoder=SAM
bsize=10000
# bsize=10000

saveto=$2

lns=$(wc -l $manifest | awk ' { print $1 }')
echo $lns

for (( i=0; i<=lns; i+=bsize ))
do
    if [ $i -lt 20000 ]; then
        continue
    fi
    echo $i
    qsub -cwd -N emb -l RTX2080Ti,gpu,cuda=1,h_rt=01:30:00 -o logs -j y \
        ./scripts/job.sh $encoder $manifest $i $bsize $saveto $3 $4 $5 $6 $7
    # break
done
