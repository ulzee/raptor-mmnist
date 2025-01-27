#%%
# %load_ext autoreload
# %autoreload 2
#%%
import torch
import torch.nn as nn
from einops import rearrange
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEConfig
from transformers import Trainer, TrainingArguments
import numpy as np
from tqdm import tqdm
from vit3d import ViTMAEForPreTraining
import argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import MM3dVolDataset, npy_npz_priority_load
import pandas as pd
#%%
class args:
    task = 'organmnist3d'
    regression = False
    multilabel = False
    device = 'cuda:0'
    lr = 1e-4
    predict = False
    pretrain_epochs = 4
    epochs = 4
    model = 'mae'
    bulkpath = None
#%%
# task = 'organmnist3d'
parser = argparse.ArgumentParser(description='Train embedding model')
parser.add_argument('--task', type=str, required=True, help='Task name')
# parser.add_argument('--model', type=str, required=True)
parser.add_argument('--batch-size', default=8, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--pretrain_epochs', default=10, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument('--predict', default=False, action='store_true')
parser.add_argument('--bulkpath', default=None, type=str)
parser.add_argument('--regression', default=False, action='store_true')
parser.add_argument('--multilabel', default=False, action='store_true')
parser.add_argument('--load-pretrained', default=False, action='store_true')

args = parser.parse_args()
args.model = 'mae'
#%%
meta = np.load(f'../medmnist/{args.task}_64.npz')
if args.regression or args.multilabel:
    num_classes = meta['test_labels'].shape[1]
else:
    num_classes = np.max(meta['test_labels']) + 1
print(f'Num classes: {num_classes}')
#%%
# Define patch size and volume size
device = torch.device(args.device)
patch_size = (16, 16, 16)  # Depth, Height, Width of each patch
if type(meta['train_images'][0]) in [str, np.str_]:
    volume_size = npy_npz_priority_load(args.bulkpath + '/' + meta['train_images'][0]).shape
else:
    volume_size = meta['train_images'][0].shape
volume_size
#%%

config = ViTMAEConfig(
    volume_size=volume_size[0],  # Replace with the flattened size of the 3D volume
    patch_size=patch_size[0],  # 3D patch size
    num_channels=1,         # Change based on your data (e.g., grayscale or multi-channel)
    decoder_hidden_size=768,
    mask_ratio=0.8,
    num_hidden_layers=4,
    # num_hidden_layers=2,
)

model = ViTMAEForPreTraining(config).to(device)
model
#%%
if not args.predict and not args.load_pretrained:
    class HFVolumeDataset(torch.utils.data.Dataset):
        def __init__(self, split, patch_size):
            self.split = split
            self.patch_size = patch_size
            self.volumes = meta[f'{self.split}_images'][:]
            self.column_names = ["pixel_values", "labels"]
            self.__version__ = '0'

        def __len__(self):
            return len(meta[f'{self.split}_labels'])

        def __getitem__(self, idx):
            volume = self.volumes[idx]

            if type(volume) in [str, np.str_]:
                volume = npy_npz_priority_load(args.bulkpath + '/' + volume)


            volume = volume[np.newaxis, :].astype(np.float32)/255
            volume = torch.from_numpy(volume)
            if volume.shape != (volume_size[0],)*3:
                volume = F.interpolate(volume.unsqueeze(0), (volume_size[0],)*3, mode='trilinear').squeeze(0)
            return {"pixel_values": volume  }

    sample_dset = HFVolumeDataset('train', patch_size)
    for sample in tqdm(sample_dset):
        pass
    #%%
    training_args = TrainingArguments(
        output_dir=f'./saved/{args.task}/mae',
        per_device_train_batch_size=4,
        num_train_epochs=args.pretrain_epochs,
        # num_train_epochs=1,
        save_total_limit=2,
        save_steps=100,
        logging_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        label_names=["pixel_values"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=HFVolumeDataset('train', patch_size),
        eval_dataset=HFVolumeDataset('val', patch_size),
    )

    trainer.train()
    #%%
    torch.save(model.state_dict(), f'saved/{args.task}/{args.model}_weights.pth')
#%%
model.load_state_dict(torch.load(f'saved/{args.task}/{args.model}_weights.pth'))
# %%
class ViTClassifier(nn.Module):
    def __init__(self, vit, hidden_dim, output_dim):
        super().__init__()
        self.vit = vit.vit
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        vit_output = self.vit(pixel_values=x)
        agg = vit_output.last_hidden_state.mean(1)
        return self.fc(agg)

cls = ViTClassifier(model, 768, num_classes).to(device)

if args.regression:
    criterion = nn.MSELoss()
elif args.multilabel:
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(cls.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs//4, gamma=0.1)

if args.predict:
    cls.load_state_dict(torch.load(f'saved/{args.task}/{args.model}_best.pth'))
# %%
datasets = { ph: MM3dVolDataset(ph, f'../medmnist/{args.task}_64.npz', bulkpath=args.bulkpath) for ph in ['train', 'val', 'test']}
loaders = { ph: DataLoader(d, batch_size=8, shuffle=ph=='train') for ph, d in datasets.items()  }

hist = []
best_val_loss = float('inf')
best_model_state = None
for epoch in range(1 if args.predict else args.epochs):
    run_phases = ['train', 'val']
    if epoch == args.epochs - 1:
        run_phases = ['train', 'val', 'test']
    if args.predict:
        run_phases = ['test']
    for ph in run_phases:
        loader = loaders[ph]

        cls.train() if ph == 'train' else cls.eval()

        pbar = tqdm(loader)
        losshist = [0, 0]
        preds = []
        optimizer.zero_grad()
        for i, data in enumerate(pbar):
            inputs = data[0].to(device).unsqueeze(1).float()/255
            if inputs.shape[-3:] != (volume_size[0],)*3:
                inputs = F.interpolate(inputs, (volume_size[0],)*3, mode='trilinear')
            if args.regression or args.multilabel:
                labels = data[1].float().to(device)
            else:
                labels = data[1].squeeze(1).to(device)

            with torch.set_grad_enabled(ph == 'train'):
                outputs = cls(inputs)
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
                torch.save(cls.state_dict() , f'saved/{args.task}/{args.model}_best.pth')
    scheduler.step()

if not args.predict: print('Finished Training')
# #%%
preds, labs = [np.concatenate(t) for t in zip(*preds)]

for colvals, saveto in zip([preds], [f'saved/{args.task}/predictions_test_{args.model}.csv']):
    dfdict = dict(ids=[f'test_{i}' for i in range(len(datasets['test']))])
    for ci, (cname, col) in enumerate(zip(range(num_classes), colvals.T)):
        dfdict[cname] = col
    pd.DataFrame(dfdict).set_index('ids').to_csv(saveto)

# %%
