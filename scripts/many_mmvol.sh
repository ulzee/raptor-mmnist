#!/bin/bash

encoder=$1

for prefix in {adrenal,nodule,synapse,fracture,organ,vessel}
do
    ./scripts/local/create_dset_script.sh $encoder $prefix
    # $prefix

    mkdir /data2/bsplat/data/embs/dset8_${encoder}_${prefix}

    parallel -j4 < ./scripts/local/by_dset/${prefix}.sh
done
