#%%
import matplotlib.pyplot as plt
import pandas as pd
import json
from tqdm import tqdm
import os, sys
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.metrics import auc as aucfn
import numpy as np

#%%
# nboots = 10
#%%
flag = ''
resolution = 64
voltag = '3d'
dsets_str = 'lidc'
# embname = 'jan9_DINO_lidc-proj_normal_d1024_k100_run1' #'dec16_Merlin_{task}-raw'
embname = 'jan9_DINO_lidc-proj_normal_d1024_k100_run1' #'dec16_Merlin_{task}-raw'
# embname = 'jan9_Merlin_lidc-avgpool' #'dec16_Merlin_{task}-raw'
# embname = 'jan17_DINO_ctrgchest-proj_normal_d1024_k100_run1' #'dec16_Merlin_{task}-raw'
#%%
dsets_str = sys.argv[1] #'dec16_Merlin_{task}-raw'
embname = sys.argv[2] #'dec16_Merlin_{task}-raw'
#%%
print(dsets_str)
print(embname)
# mdl_options = sys.argv[3]
# knum = 100
# embname = 'dec15_DINO_{task}-proj_normal_d1024_k%d_contr_run1' % knum
# embname = 'dec14_DINO_{task}-proj_normal_d1024_k%d_run3' % knum
# embname = 'dec16_VoCo_{task}-avg_cat'
# embname = 'dec16_VoCo_{task}-planar_avg_cat'
# probe = f'logr{mdl_options}-l2'
probe = f'logr-l2'
lrs = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
# lrs = [0.0000001, 0.000001]
# if 'dec15' in embname or 'dec14' in embname:
#     lrs += []
# embname = 'dec9_DINO_{task}-proj_normal_d1024_k%d_run3' % knum
# embname = 'dec9_SAM_{task}-proj_normal_d256_k%d_run3' % knum
# for task in ['adrenal', 'vessel']:
# for task in ['fracture']:
dsets = dsets_str.split(',') #['adrenal', 'fracture', 'nodule', 'organ', 'synapse', 'vessel']
dfolders = []
if 'simulated' in embname:
    dfolders = ['nodule']
else:
    dfolders = [f'{task}mnist{voltag}' for task in dsets]
for dfl, task in zip(dfolders, dsets):
    load_emb = embname.replace('DATASET', task)
    mdls = [
        # f'saved/{task}mnist{voltag}/predictions_[split]_{probe}_best{flag}_dec3_{task}-proj_normal_k100.csv',
        f'saved/{dfl}/predictions_[split]_{probe}_best{flag}_{load_emb}.csv',
    ]

    # if mdl_options:
    #     mdls[0] = mdls[0].replace('_best', f'-{mdl_options}_best')
    # print(mdls)
    # assert False

    preds_by_split = dict()
    for split in ['train', 'val', 'test']:

        preds_by_split[split] = []

        for mdl in mdls:
            mdl = mdl.replace('[split]', split)
            mname = mdl.split(f'predictions_{split}_')[-1].split('.')[0]
            bylr = []
            for lr in lrs:
                if lr is None:
                    load_file = mdl.replace('-l2', '')
                    bylr += [(lr, pd.read_csv(load_file).set_index('ids'))]
                else:
                    bylr += [(lr, pd.read_csv(mdl.replace('-l2', f'-l2-a{lr}')).set_index('ids'))]
            preds_by_split[split] += [(mname, bylr)]

    def main_bar(x, ls):
        return plt.boxplot([ls], positions=[x], widths=[0.5], patch_artist=True)

    def base_bar(x, ls):
        h = np.median(ls)
        plt.boxplot([ls], positions=[x], widths=[0.5])
        plt.axhline(h, color=f'C{x}', linestyle='dashed')

    # evalfn = lambda a, b: np.corrcoef(a, b)[0, 1]**2
    # evalfn = average_precision_score
    evalfn = roc_auc_score
    # if 'best_nocov' in mdls[0]:
    #     evalfn = lambda a, b: np.corrcoef(a, b)[0, 1]**2
    print('Eval', evalfn)

    plt.figure(figsize=(12, 4))


    test_scores = dict()
    best_test_by_val = []
    targets_by_split = dict()
    # for ci, c in enumerate(target_all.columns):
    if 'simulated' in load_emb:
        task = 'nodule'
        simtask = load_emb.split('simulated_')[1].split('_')[0]
        ldf = np.load(f'../medmnist/simulated/{simtask}/test_labels.npy')

        # if 'location' in load_emb:
        #     cls_idx = [4, 8, 16, 32, 64, 128].index(int(mdl_options.split('etc')[1]))
        #     ldf = np.array([[[0, 1][int(lb[0] == cls_idx)]] for lb in ldf])
    else:
        ldf = np.load(f'../medmnist/{task}mnist{voltag}_{resolution}.npz')[f'test_labels']
    if ldf.shape[1] == 1:
        nclasses = int(np.max(ldf) + 1)
    else:
        nclasses = ldf.shape[1]

    print('Nclases:', nclasses)

    # print(bylr[0][1].shape)
    # assert False

    ls_by_class = []
    for ci, c in enumerate(range(nclasses)):
        test_scores[ci] = dict()

        best_test_by_val += [[None for _ in preds_by_split['test']]]

        plt.subplot(1, nclasses, ci+1)
        # c = target_all.columns[0]
        styles = [ 'dotted', 'dashed', 'solid' ]
        # styles = [ 'dotted', 'solid' ]
        # for si, split in enumerate(['train', 'test']):

        for si, split in enumerate(['val', 'test']):
        # for si, split in enumerate(['train', 'val', 'test']):

            if 'simulated' in load_emb:
                task = 'nodule'
                simtask = load_emb.split('simulated_')[1].split('_')[0]
                ldf = np.load(f'../medmnist/simulated/{simtask}/{split}_labels.npy')

                # if 'location' in load_emb:
                #     cls_idx = [4, 8, 16, 32, 64, 128].index(int(mdl_options.split('etc')[1]))
                #     ldf = np.array([[[0, 1][int(lb[0] == cls_idx)]] for lb in ldf])
            else:
                ldf = np.load(f'../medmnist/{task}mnist{voltag}_{resolution}.npz')[f'{split}_labels']

            if ldf.shape[1] > 1:
                ldfmat = ldf # it is already a onehot of multi labels
            else:
                ldfmat = np.zeros((len(ldf), nclasses))
                for i, li in enumerate(ldf):
                    ldfmat[i, li] = 1

            labels = ldfmat[:, ci]

            # targets_by_split[split] = resids

            # boot_ixs = np.load(f'saved/icd/boots10_{split}.npy')

            for mi, (mdl, bylr) in enumerate(preds_by_split[split]):
                test_scores[ci][mi] = dict()

                ls = []
                for lr, pdf in bylr:
                    reps = []
                    # for bls in boot_ixs:
                    #     # bls = bls[:nboots]
                    #     reps += [evalfn(resids.loc[pdf.index][c].values[bls], pdf[c].values[bls])]
                    # if lr == None and split == 'test':
                    #     plt.axhline(np.mean(reps), color=f'C{mi}', alpha=0.3)
                    # else:
                    preds = pdf[str(c)].values
                    if np.isnan(np.sum(labels)): print('Label has nans')
                    if np.isnan(np.sum(preds)): print('Preds have nans')
                    est = evalfn(labels, preds)

                    ls += [est]
                    # if split == 'test':
                    #     test_scores[ci][mi][lr] = [np.mean(reps), reps]

                    # if split == 'val':
                    #     if lr is None: continue
                    #     if best_test_by_val[ci][mi] is None or np.mean(reps) > best_test_by_val[ci][mi][0]:
                    #         # keep the test scores
                    #         best_test_by_val[ci][mi] = [np.mean(reps), lr, None]

                plt.plot(
                    range(len(ls)),
                    ls,
                    color=f'C{mi}',
                    linestyle=styles[si],
                    marker='o', label=split)

                if split == 'val':
                    ls_by_class += [ls]

        plt.xticks(range(len(lrs)), lrs)
        plt.legend()


    best_ix = np.argmax(np.mean(ls_by_class, axis=0))
    print('Best alpha:', lrs[best_ix], np.mean(ls_by_class, axis=0)[best_ix])

    saveto = f'saved/{task}mnist{voltag}/formatted'

    os.makedirs(saveto, exist_ok=True)


    mdl_name = mdls[0].split('[split]_')[1].split('.')[0]

    save_name = f'{saveto}/{task}mnist{voltag}_{resolution}_test@{mdl_name}.csv'

    savedf = preds_by_split['test'][0][1][best_ix][1].copy()
    savedf.index = range(len(savedf))
    savedf.to_csv(save_name, header=None)
    print(save_name)
    save_name

    save_name

# %%
