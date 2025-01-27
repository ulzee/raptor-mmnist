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

probe = f'logr-l2'
lrs = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
dsets = dsets_str.split(',') #['adrenal', 'fracture', 'nodule', 'organ', 'synapse', 'vessel']

dfolders = []
if 'simulated' in embname:
    dfolders = ['nodule']
else:
    dfolders = [f'{task}mnist{voltag}' for task in dsets]
for dfl, task in zip(dfolders, dsets):
    load_emb = embname.replace('DATASET', task)
    mdls = [
        f'saved/{dfl}/predictions_[split]_{probe}_best{flag}_{load_emb}.csv',
    ]

    if len(sys.argv) > 3:
        mdls[0] = mdls[0].replace('_best', f'-{sys.argv[3]}_best')


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

    else:
        ldf = np.load(f'../medmnist/{task}mnist{voltag}_{resolution}.npz')[f'test_labels']
    if ldf.shape[1] == 1:
        nclasses = int(np.max(ldf) + 1)
    else:
        nclasses = ldf.shape[1]

    print('Nclases:', nclasses)

    ls_by_class = []
    for ci, c in enumerate(range(nclasses)):
        test_scores[ci] = dict()

        best_test_by_val += [[None for _ in preds_by_split['test']]]

        plt.subplot(1, nclasses, ci+1)
        # c = target_all.columns[0]
        styles = [ 'dotted', 'dashed', 'solid' ]

        for si, split in enumerate(['val', 'test']):
        # for si, split in enumerate(['train', 'val', 'test']):

            if 'simulated' in load_emb:
                task = 'nodule'
                simtask = load_emb.split('simulated_')[1].split('_')[0]
                ldf = np.load(f'../medmnist/simulated/{simtask}/{split}_labels.npy')

            else:
                ldf = np.load(f'../medmnist/{task}mnist{voltag}_{resolution}.npz')[f'{split}_labels']

            if ldf.shape[1] > 1:
                ldfmat = ldf # it is already a onehot of multi labels
            else:
                ldfmat = np.zeros((len(ldf), nclasses))
                for i, li in enumerate(ldf):
                    ldfmat[i, li] = 1

            labels = ldfmat[:, ci]

            for mi, (mdl, bylr) in enumerate(preds_by_split[split]):
                test_scores[ci][mi] = dict()

                ls = []
                for lr, pdf in bylr:
                    reps = []

                    preds = pdf[str(c)].values
                    if np.isnan(np.sum(labels)): print('Label has nans')
                    if np.isnan(np.sum(preds)): print('Preds have nans')
                    est = evalfn(labels, preds)

                    ls += [est]

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
