#%%
# %load_ext autoreload
# %autoreload 2
#%%
import json
import pandas as pd
import numpy as np
import pickle as pk
import torch
import torch.nn as nn
import torch.optim as optim
# Load and transform the dataset
import os, sys
# from torch.utils.data import Dataset
from nns import MLP
from datasets import MM3DEmbDataset, MM3DEmbSimDataset, MM3DRetrievalDataset
from sklearn.multiclass import OneVsRestClassifier
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet, Lasso
#%%
class args:
    model = 'logr'
    # task = 'shape'
    task = 'location'
    # emb_type = 'dec14_DINO_nodule/proj_normal_d1024_k100_run1'
    # emb_type = 'dec18_simulated_shape_DINO_nodule/proj_normal_d1024_k100_run1'
    emb_type = 'dec18_simulated_shape_s64_DINO_nodule/proj_normal_d1024_k100_run1'
    penalty = 'l2'
    alpha = 0.0001
    dropout = None
    # weight_decay = None
    disable_normalize = False
    planes = None
    save_all_splits = True
    etc = '4'
#%%
import argparse

parser = argparse.ArgumentParser(description='Train embedding model')
# parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--task', type=str, required=True, help='Task name')
parser.add_argument('--emb-type', type=str, default=None, help='Embedding type')
parser.add_argument('--model', type=str, default='logr')
parser.add_argument('--save_all_splits', action='store_true', default=False)
parser.add_argument('--penalty', default=None)
parser.add_argument('--alpha', default=None, type=float)
parser.add_argument('--weight-decay', default=None, type=float)
parser.add_argument('--dropout', default=None, type=float)
parser.add_argument('--disable-normalize', default=False, action='store_true')
parser.add_argument('--planes', default=None, type=str)
parser.add_argument('--layers', default=1, type=int)
parser.add_argument('--temperature', default=None, type=float)
parser.add_argument('--etc', default=None, type=str)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--sub', default=None, type=int)

args = parser.parse_args()
#%%
emb_type = args.emb_type
task = args.task
embname = emb_type.replace('/', '-')
if args.planes is not None:
    args.planes = args.planes.split(',')

phases = ['train', 'val', 'test']
imgs_dir = '../medmnist'
embs_dir = f'../data/embs/{emb_type}'
# task_name = task if 'icd' not in task else ('icd/'+task.split('/')[-1].split('.csv')[0])

pflag = ''
if args.planes is not None:
    pflag += f'-pln{"".join(args.planes)}'
if args.penalty is not None:
    pflag += f'-{args.penalty}'
if args.alpha is not None:
    pflag += f'-a{args.alpha}'
if args.etc is not None:
    pflag += f'-etc{args.etc}'
if args.sub is not None:
    pflag += f'-sub{args.sub}'

if args.model == 'retr':
    pflag += f'-l{args.layers}'
    pflag += f'-ep{args.epochs}'
    if args.weight_decay is not None:
        pflag += f'-w{args.weight_decay}'
    if args.dropout is not None:
        pflag += f'-d{args.dropout}'
    if args.temperature is not None:
        pflag += f'-t{args.temperature}'

# if args.nocov:
#     savename = f'{args.model}{pflag}_best_nocov_{embname}'
# else:
savename = f'{args.model}{pflag}_best_{embname}'
task_name = args.task.split("_")[0]
if args.sub is None:
    manifest = np.genfromtxt(f'saved/manifests/{task_name}.txt', dtype=str)
else:
    manifest = np.genfromtxt(f'saved/manifests/{task_name}_sub{args.sub}.txt', dtype=str)
phase_ids = dict()
for ph in phases:
    phase_ids[ph] = [i for i in manifest if ph in i]
print(savename)
#%%
# class EmbDataset(Dataset):
#     def __init__(self, split, task, root_dir, emb_dir, split_ids, stats=None):

#         self.root_dir = root_dir
#         self.emb_dir = emb_dir
#         self.split = split
#         self.stats = stats
#         self.split_ids = split_ids

#         blob = np.load(f'{root_dir}/{task}.npz')
#         self.labels = blob[f'{split}_labels'] # NOTE: make sure to index correctly if using subsamples

#     def __len__(self):
#         return len(self.split_ids)

#     def __getitem__(self, index):

#         pid = self.split_ids[index]
#         pix = int(pid.split('_')[1])

#         evec = np.load(f'{self.emb_dir}/{pid}.npz')['arr_0']
#         return evec, self.labels[pix]

dsetclass = MM3DEmbDataset
if args.task in ['location', 'shape']:
    dsetclass = MM3DEmbSimDataset
if args.model in ['retr']:
    dsetclass = MM3DRetrievalDataset
# split, task, labels_dir, embs_dir, split_ids
trainset = dsetclass(split='train', task=task, labels_dir=imgs_dir, embs_dir=embs_dir, split_ids=phase_ids['train'], args=args)
dsets = dict(
    train=trainset,
    val=dsetclass(split='val', task=task, labels_dir=imgs_dir, embs_dir=embs_dir, split_ids=phase_ids['val'], stats=trainset.stats, args=args),
    test=dsetclass(split='test', task=task, labels_dir=imgs_dir, embs_dir=embs_dir, split_ids=phase_ids['test'], stats=trainset.stats, args=args),
)
#%%
if args.model != 'retr':
    train_x_mat, train_y = map(np.stack, zip(*[s for s in tqdm(dsets['train'])]))
    val_x_mat, val_y = map(np.stack, zip(*[s for s in tqdm(dsets['val'])]))
    test_x_mat, _ = map(np.stack, zip(*[s for s in tqdm(dsets['test'])]))

    if train_y.shape[1] == 1:
        train_y = train_y[:, 0]
    if val_y.shape[1] == 1:
        val_y = val_y[:, 0]

    # nclasses = int(np.max(train_y) + 1)
    data_mats = dict(train=train_x_mat, val=val_x_mat, test=test_x_mat)
#%%

if args.planes is not None:
    print('Choosing planes:', train_x_mat.shape)
    assert train_x_mat.shape[-1] % 3 == 0
    plane_size = train_x_mat.shape[-1]//3

    byplane = [list() for _ in range(3)]
    for plane_i, pname in enumerate(['A', 'C', 'S']):
        if pname in args.planes:
            for si, xmat in enumerate([train_x_mat, val_x_mat, test_x_mat]):
                byplane[si] += [xmat[:, plane_size*plane_i:plane_size*(plane_i+1)]]
    byplane = [np.concatenate(pls, -1) for pls in byplane]
    train_x_mat, val_x_mat, test_x_mat = byplane
    print('Using planes    :', train_x_mat.shape)

#%%

save_results = dict()

for phase in phases:

    if args.model == 'logr':
        # simple logr for binary classification
        if phase == 'train':
            print('Model:', 'LogisticRegression', args.penalty, args.alpha)
            lroptions = dict(random_state=0)
            lroptions['penalty'] = None
            if args.penalty:
                lroptions['penalty'] = args.penalty
            if args.alpha:
                lroptions['C'] = args.alpha

            mdl = LogisticRegression(**lroptions)
            if len(train_y.shape) > 1:
                mdl = OneVsRestClassifier(mdl)
            mdl.fit(train_x_mat, train_y)

        pred_allvars = mdl.predict_proba(data_mats[phase])

    elif args.model in ['retr']:
        pred_allvars = None
        if phase == 'train':
            device = torch.device('cuda:2')
            output_dim = len(dsets['train'][0][1])
            mconfig = dict(
                # input_dim=output_dim,
                hidden_dim=output_dim,
                # hidden_dim=output_dim//2,
                input_dim=len(dsets['train'][0][0]),
                output_dim=output_dim,
                num_layers=args.layers)
            if args.dropout is not None: mconfig['dropout'] = args.dropout
            model = MLP(**mconfig).to(device)
            criterion = nn.MSELoss()
            optargs = dict()
            optimizer = optim.AdamW(model.parameters(), lr=0.0001, **optargs)
            # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs//4, gamma=0.5)
            print(model)

            dloaders = { ph: torch.utils.data.DataLoader(
                dsets[ph], batch_size=2048, shuffle=ph=='train', drop_last=ph=='train') for ph in phases }
            best_val_loss = None
            # for epoch in range(400):
            for epoch in range(args.epochs):
                # for inner_phase in ['train']:
                for inner_phase in ['train', 'val']:
                    model.train() if inner_phase == 'train' else model.eval()

                    pbar = tqdm(dloaders[inner_phase])
                    loss_hist = []
                    top1, top5 = 0, 0
                    for batch in pbar:
                        x, y = [t.to(device).float() for t in batch]

                        with torch.set_grad_enabled(inner_phase == 'train'):
                            if inner_phase == 'train': optimizer.zero_grad()
                            outputs = model(x)
                            e0 = F.normalize(outputs, p=2, dim=-1)
                            e1 = F.normalize(model.compare_bn(y), p=2, dim=-1)

                            similarity = torch.matmul(e0, e1.T)
                            # # Example of contrastive loss
                            def contrastive_loss(similarity, temperature=0.1):
                                if temperature is None: temperature = 0.1
                                # Apply temperature scaling
                                similarity = similarity / temperature

                                # Labels for positive pairs
                                batch_size = similarity.size(0)
                                labels = torch.arange(batch_size).to(similarity.device)

                                # Cross-entropy loss for alignment
                                loss_image_to_text = F.cross_entropy(similarity, labels)
                                loss_text_to_image = F.cross_entropy(similarity.T, labels)
                                return (loss_image_to_text + loss_text_to_image) / 2

                            # loss = contrastive_loss(similarity, args.temperature)

                            loss = ((((e0 - e1)**2).sum(-1))**0.5).mean()

                            if inner_phase == 'train':
                                loss.backward()
                                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                                optimizer.step()

                            similarity = similarity.detach().cpu().numpy()
                            for si, sim in enumerate(similarity):
                                srank = np.argsort(sim)[::-1]
                                if si == srank[0]: top1 += 1
                                if si in srank[:5]: top5 += 1

                        loss_hist += [loss.item()]

                        pbar.set_postfix(dict(ep=epoch, l='%.4f'%np.mean(loss_hist[-20:]), top1=top1, top5=top5))
                    # break
                # print((((e0[0]-e1[0])**2).sum()**0.5).item())
                # scheduler.step()

                # if best_val_loss is None or np.mean(loss_hist) < best_val_loss:
                #     best_val_loss = np.mean(loss_hist)
                torch.save(model.state_dict(), f'cache/{savename}.pth')
        # if phase == 'test':
        model.load_state_dict(torch.load(f'cache/{savename}.pth', weights_only=True))
        predloader = torch.utils.data.DataLoader(dsets[phase], batch_size=32, shuffle=False)
        pred_allvars = np.concatenate([model(x.float().to(device)).detach().cpu().numpy() for x, _ in tqdm(predloader)])
    else:
        raise 'Not implemented'

    save_results[phase] = pred_allvars
# print(pred_allvars)
#%%
if args.model in ['retr']:
    os.makedirs(f'saved/{task_name}', exist_ok=True)
    for ph in phases:
        np.save(f'saved/{task_name}/queries_{ph}_{savename}.npy', save_results[ph])
else:
    save_splits = ['train', 'val', 'test'] if args.save_all_splits else ['test']
    print('Saving:', save_splits)
    for split in save_splits:
        if 'simulated' in args.emb_type:
            task_name = 'nodule'
        for colvals, saveto in zip([save_results[split]], [f'saved/{task_name}/predictions_{split}_{savename}.csv']):
            # print(colvals.shape)
            os.makedirs(os.path.dirname(saveto), exist_ok=True)

            dfdict = dict(ids=dsets[split].split_ids)
            for ci, (cname, col) in enumerate(zip(range(colvals.shape[1]), colvals.T)):
                # print(col.shape)
                # if trainset.stats is not None:
                #     dfdict[cname] = (col*trainset.stats[1].values[ci]) + trainset.stats[0].values[ci]
                # else:
                dfdict[cname] = col
            pd.DataFrame(dfdict).set_index('ids').to_csv(saveto)
            print(colvals.shape)
            print(saveto)
