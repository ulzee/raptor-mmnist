#!/bin/bash

mdl=logr-l2
embname=dec3_dset-proj_normal_k100.

# for dset in
# for dset in {organ,nodule,fracture,adrenal,vessel,synapse}
# do
#     echo $dset
#     embname_wdset=${embname/dset/$dset}
#     fname=${dset}mnist3d_64_test@${mdl}_best_${embname_wdset}csv
#     scorefile=saved/${dset}mnist3d/formatted/$fname
#     echo $scorefile
#     python -m medmnist evaluate --path=$scorefile
# done

for dset in {organ,nodule,fracture,adrenal,vessel,synapse}
do

    embname_wdset=${embname/dset/$dset}
    for fname in saved/${dset}mnist3d/formatted/*AUC*${mdl}*${embname_wdset}*
    do
        # echo $fname
        echo $dset
        auc=$(echo $fname | grep -oP 'AUC\]\K[0-9]+\.[0-9]+')
        acc=$(echo $fname | grep -oP 'ACC\]\K[0-9]+\.[0-9]+')
        echo "${auc} & ${acc}"
    done

done
