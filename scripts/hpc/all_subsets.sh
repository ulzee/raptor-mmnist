#!/bin/bash

for sub in {10,50,100,200,500}
do
    # ./scripts/hpc/resnet50_all.sh synapse "--sub $sub"
    ./scripts/hpc/pctnet_all.sh synapse "--sub $sub"
    ./scripts/hpc/suprem_all.sh synapse "--sub $sub"
done
