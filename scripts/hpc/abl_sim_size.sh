#!/bin/bash

embname=dec22_simulated_shape_RES_DINO_nodule/proj_normal_d1024_k100_run1
mode=$1
if [ "$mode" == "tune" ]; then

    # for RES in {64,32,16,8}
    for RES in 16
    do
        for run in {1,2,3}
        do
            use_emb=${embname/run1/"run$run"}
            use_emb=${use_emb/RES/"s$RES"}

            qsub -cwd -t 1-8:1 \
                -N sim-shp-tune -l h_data=8G,h_rt=00:30:00 -pe shared 4 -o logs -j y \
                ./scripts/hpc/tune_job.sh shape $use_emb
            # break
        done
        # break
    done
else
    for RES in {64,32,16,8}
    do
        for run in {1,2,3}
        do
            use_emb=${embname/run1/"run$run"}
            use_emb=${use_emb/RES/"s$RES"}
            use_emb=${use_emb/"/"/"-"}

            python save_best_l2_crossval.py $use_emb ""
        done
    done
fi
