#!/bin/bash

embname=dec14_DINO_DATASET/proj_normal_d1024_K_run1

mode=$1

# for k in {1,5,10,50,100}
# for k in {10,100,150}
# for k in {1,5,50}

if [ "$mode" == "tune" ]; then
    # for k in {1,5,10,100,150}
    for k in 5
    do
        # for run in {2,3}
        for run in {1,2,3}
        do
            # for dset in synapse
            for dset in {adrenal,fracture,nodule,organ,synapse,vessel}
            do
                use_emb=${embname/DATASET/$dset}
                use_emb=${use_emb/K/"k$k"}
                use_emb=${use_emb/run1/"run$run"}

                qsub -cwd -t 1-8:1 \
                    -N tune-${dset} -l h_data=8,h_rt=00:30:00 -pe shared 4 -o logs -j y \
                    ./scripts/hpc/tune_job.sh $dset $use_emb
                # break
            done
            # break
        done
        # break
    done
else
    embname=${embname/"/"/"-"}
    # after everything finishess...
    # for k in {10,100,150}
    for k in {1,5}
    do
        for run in {1,2,3}
        do
            use_emb=${embname/K/"k$k"}
            use_emb=${use_emb/run1/"run$run"}
            python save_best_l2_crossval.py $use_emb ""
        done
    done
fi


