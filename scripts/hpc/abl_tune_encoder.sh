#!/bin/bash

embname=dec24_ENCODER_DATASET/proj_normal_LATENT_k100_run1

mode=$1

latents=(256 256 1024)

mi=0
if [ "$mode" == "tune" ]; then
    # for encoder in CLIP
    for encoder in {SAM,MedSAM,CLIP}
    do
        # for run in 3
        for run in {1,2,3}
        do
            # for dset in vessel
            for dset in {adrenal,fracture,nodule,organ,synapse,vessel}
            do
                use_emb=${embname/DATASET/$dset}
                use_emb=${use_emb/ENCODER/$encoder}
                use_emb=${use_emb/run1/"run$run"}
                use_emb=${use_emb/LATENT/"d${latents[$((mi))]}"}
                # use_emb=${use_emb/LATENT/d1024}


                echo $use_emb
                qsub -cwd -t 1-8:1 \
                    -N tune-$encoder-${dset} -l h_data=8,h_rt=00:30:00 -pe shared 4 -o logs -j y \
                    ./scripts/hpc/tune_job.sh $dset $use_emb
                # break
            done
            # break
        done
        ((mi++))
        # break
    done
else
    embname=${embname/"/"/"-"}
    # for encoder in CLIP
    for encoder in {SAM,MedSAM,CLIP}
    do
        for run in {1,2,3}
        do
            # for dset in vessel
            for dset in {adrenal,fracture,nodule,organ,synapse,vessel}
            do
                # use_emb=${embname/DATASET/$dset}
                use_emb=${embname/ENCODER/$encoder}
                use_emb=${use_emb/run1/"run$run"}
                use_emb=${use_emb/LATENT/"d${latents[$((mi))]}"}
                # use_emb=${use_emb/LATENT/d1024}

                echo $use_emb
                python save_best_l2_crossval.py $use_emb ""
            done
        done
        ((mi++))
    done
fi


