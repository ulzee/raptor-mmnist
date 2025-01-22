#%%
import argparse
import json
import pandas as pd
import numpy as np
import pickle as pk
import torch
import torch.nn as nn
import torch.optim as optim
import os, sys
from nns import MLP
from datasets import MM3DEmbDataset, MM3DEmbSimDataset, MM3DRetrievalDataset, MM3dVolDataset
from sklearn.multiclass import OneVsRestClassifier
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet, Lasso
from torch.utils.data import Dataset, DataLoader
#%%

parser = argparse.ArgumentParser(description='Train embedding model')
parser.add_argument('--task', type=str, required=True, help='Task name')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--batch-size', default=8, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument('--predict', default=False, action='store_true')
parser.add_argument('--bulkpath', default=None, type=str)
parser.add_argument('--regression', default=False, action='store_true')
parser.add_argument('--multilabel', default=False, action='store_true')

args = parser.parse_args()
#%%
device = torch.device(args.device)
blob = np.load(f'../medmnist/{args.task}_64.npz')
if args.regression or args.multilabel:
    num_classes = blob['test_labels'].shape[1]
else:
    num_classes = np.max(blob['test_labels']) + 1
print(f'Num classes: {num_classes}')
#%%
if args.model == 'resnet50':
    from baselines.models import resnet3d50 as Resnet3d

    input_size = (224, 224, 224)

    model = Resnet3d(num_classes=num_classes).to(device)

elif args.model == 'pctnet':
    input_size = (64, 128, 128)

    sys.path.append('../etc/MIS-FM/')
    from net.pct_net import PCTNet
    from pymic.util.parse_config import parse_config
    from pymic.util.general import mixup, tensor_shape_match

    class defaults:
        stage = 'train'
        cfg = '../etc/MIS-FM/demo/pctnet_pretrain.cfg'
        num_classes = 100 # infer a sufficient number of outplanes for most tasks

    config = parse_config(defaults)
    config['network']['class_num'] = defaults.num_classes

    class PCTNetClassifier(PCTNet):
        def __init__(self, params):
            super().__init__(params)

            self.fc = nn.Linear(128+defaults.num_classes*2, num_classes)

        def forward(self, x):
            x0, x1, x2 = self.embeddings[self.resolution_mode](x)
            x2  = self.pyramid_ct(x2)
            pooled = [F.adaptive_avg_pool3d(s, (1, 1, 1)).view(len(s), -1) for s in x2]
            pooled = torch.cat(pooled, -1)
            return self.fc(pooled)

    model = PCTNetClassifier(config['network']).to(device)

    def load_pretrained_weights(network, pretrained_dict, device_ids):
        if(len(device_ids) > 1):
            if(hasattr(network.module, "get_parameters_to_load")):
                model_dict = network.module.get_parameters_to_load()
            else:
                model_dict = network.module.state_dict()
        else:
            if(hasattr(network, "get_parameters_to_load")):
                model_dict = network.get_parameters_to_load()
            else:
                model_dict = network.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if \
            k in model_dict and tensor_shape_match(pretrained_dict[k], model_dict[k])}
        if (len(device_ids) > 1):
            network.module.load_state_dict(pretrained_dict, strict = False)
        else:
            network.load_state_dict(pretrained_dict, strict = False)
    w = torch.load('../etc/MIS-FM/weights/pctnet_ct10k_volf.pt')['model_state_dict']
    load_pretrained_weights(model, w, [None])
elif args.model == 'suprem':
    from monai.networks.nets import SegResNet

    input_size = (96,)*3

    def simplify_key(k):
        for prefix in ['module.', 'features.', 'backbone.', 'model.']:
            k = k.replace(prefix, '')
        return k

    class defaults:
        segresnet_init_filters = 16

    class SegResNetClassifier(SegResNet):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            self.fc = nn.Linear(128, kwargs['out_channels'])

        def forward(self, x):
            x, down_x = self.encode(x)
            return self.fc(F.adaptive_avg_pool3d(x, (1, 1, 1)).view(len(x), 128))

    model = SegResNetClassifier(
                blocks_down=[1, 2, 2, 4],
                blocks_up=[1, 1, 1],
                init_filters=defaults.segresnet_init_filters,
                in_channels=1,
                out_channels=num_classes,
                dropout_prob=0.0,
                )

    weights = 'pretrained_weights/supervised_suprem_segresnet_2100.pth'
    model_dict = torch.load('../etc/SuPreM/target_applications/pancreas_tumor_detection/' + weights, weights_only=True)['net']
    store_dict = model.state_dict()
    simplified_model_dict = {simplify_key(k): v for k, v in model_dict.items()}
    amount = 0
    for key in store_dict.keys():
        if key in simplified_model_dict and 'conv_final.2.conv' not in key:
            store_dict[key] = simplified_model_dict[key]
            amount += 1
    # assert amount == (len(store_dict.keys())-2), 'the pre-trained model is not loaded successfully'
    print('loading weights', amount, len(store_dict.keys()))
    model.load_state_dict(store_dict, strict=False)

    model = model.to(args.device)
else:
    raise 'Not implemented'

datasets = { ph: MM3dVolDataset(ph, f'../medmnist/{args.task}_64.npz', args.bulkpath) for ph in ['train', 'val', 'test'] }
loaders = { ph: DataLoader(d, batch_size=args.batch_size, shuffle=ph=='train') for ph, d in datasets.items() }
for ph, d in datasets.items():
    print(ph, len(d))

if args.regression:
    criterion = nn.MSELoss()
elif args.multilabel:
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=args.lr)

if args.predict:
    model.load_state_dict(torch.load(f'saved/{args.task}/{args.model}_best.pth'))

hist = []
best_val_loss = float('inf')
best_model_state = None
for epoch in range(1 if args.predict else args.epochs):
    for ph in ['test'] if args.predict else ['train', 'val', 'test']:
        loader = loaders[ph]

        model.train() if ph == 'train' else model.eval()

        pbar = tqdm(loader)
        losshist = [0, 0]
        preds = []
        optimizer.zero_grad()
        for i, data in enumerate(pbar):
            inputs = data[0].to(device).unsqueeze(1).float()/255
            if inputs.shape[-3:] != input_size:
                inputs = F.interpolate(inputs, input_size, mode='trilinear')
            if args.regression or args.multilabel:
                labels = data[1].float().to(device)
            else:
                labels = data[1].squeeze(1).to(device)

            with torch.set_grad_enabled(ph == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if ph == 'train':
                    loss.backward()
                    # if (i + 1) % 10 == 0 or i == len(pbar) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                losshist[0] += loss.item()*len(inputs)
                losshist[1] += len(inputs)

            if ph == 'test':
                #FIXME: sigmoid? for multi label setting
                if args.regression:
                    pass
                elif args.multilabel:
                    outputs = torch.sigmoid(outputs)
                else:
                    outputs = torch.softmax(outputs, dim=-1)
                outputs = outputs.cpu().numpy()
                preds += [(outputs, labels.cpu().numpy())]

            pbar.set_postfix(dict(e=epoch, p=ph, ls='%.4f'%(losshist[0]/losshist[1])))

        if not args.predict:
            if ph == 'val' and losshist[0]/losshist[1] < best_val_loss:
                best_val_loss = losshist[0]/losshist[1]
                torch.save(model.state_dict() , f'saved/{args.task}/{args.model}_best.pth')

if not args.predict: print('Finished Training')
# #%%
preds, labs = [np.concatenate(t) for t in zip(*preds)]

for colvals, saveto in zip([preds], [f'saved/{args.task}/predictions_test_{args.model}.csv']):
    dfdict = dict(ids=[f'test_{i}' for i in range(len(datasets['test']))])
    for ci, (cname, col) in enumerate(zip(range(num_classes), colvals.T)):
        dfdict[cname] = col
    pd.DataFrame(dfdict).set_index('ids').to_csv(saveto)
