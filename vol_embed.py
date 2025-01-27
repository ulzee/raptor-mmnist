#%%
import os, sys
import numpy as np
from tqdm import tqdm
import torch
import torchvision
import copy
import argparse

from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
#%%
class args:
    dataroot = '/data2/medmnist/nodulemnist3d_64.npz'
    encoder = 'VoCo'
    manifest = 'saved/manifests/nodulemnist3d.txt'
    device = 'cuda:0'
    start = 0
    many = 5
    batch_size = 64
    saveto = '/data2/bsplat/data/embs/dec17_Merlin_nodule'
    k = 10
    planar = True
    avgpool = False
#%%
parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--dataroot', type=str, default='/u/project/sgss/UKBB/imaging/bulk/20253', help='Data root directory')
parser.add_argument('--encoder', type=str, default='VoCo', help='Encoder type')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
parser.add_argument('--manifest', type=str, required=True, help='Manifest file path')
parser.add_argument('--start', type=int, required=True, help='Start index')
parser.add_argument('--many', type=int, required=True, help='Number of files to process')
parser.add_argument('--saveto', type=str, required=True, help='Save directory')
parser.add_argument('--k', type=int, default=None)
parser.add_argument('--avgpool', action='store_true', default=False)
parser.add_argument('--planar', action='store_true', default=False)
# parser.add_argument('--raw', action='store_true', default=False)
parser.add_argument('--batch_size', type=int, default=-1, help='(unused) Batch size')
parser.add_argument('--all_slices', action='store_true', default=False, help='(unused)')
# parser.add_argument('--save_sam', action='store_true', default=False)

args = parser.parse_args()
#%%
# if args.k is not None:
#     args.k = [int(i) for i in args.k.split(',')]
# assert args.avgpool or args.planar
if args.avgpool:
    assert args.k is None # there is no need for Ks
#%%

with open(args.manifest) as fl:
    fls = [ln.strip() for ln in fl if ln]
fbatch = fls[args.start:args.start+args.many]
fbatch
#%%

input_volume_size = dict(
    VoCo=96,
    Merlin=224,
)[args.encoder]
latent_size = dict(
    VoCo=384,
    Merlin=2048,
)[args.encoder]

if args.encoder in ['VoCo']:
    sys.path.append('../etc/LSM/VoComni')
    from models import VoCo

    class voco_default_args:
        roi_x = 96
        roi_y = 96
        roi_z = 96
        in_channels = 1
        out_channels = 21
        feature_size = 192
        use_ssl_pretrained = True
        dropout_path_rate = 0
        use_checkpoint = None
        pretrained_root = '../etc/LSM/pretrained'

    voco_model = VoCo(voco_default_args).to(args.device).eval()
    emb_model = voco_model.swinViT
elif args.encoder in ['Merlin']:
    sys.path.append('../etc/Merlin')
    sys.path.append('../etc/Merlin/merlin')
    import torch.utils.checkpoint as checkpoint
    from models.load import Merlin
    from models.i3res import I3ResNet

    class I3ResnetEncoder(I3ResNet):
        def __init__(self, resnet2d, **kwargs):
            super().__init__(resnet2d, **kwargs)

        def encode(self, x):
            skips = []
            x = x.permute(0, 1, 4, 2, 3)
            x = torch.cat((x, x, x), dim=1)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = checkpoint.checkpoint(self.layer1, x)
            x = checkpoint.checkpoint(self.layer2, x)
            x = checkpoint.checkpoint(self.layer3, x)
            x = checkpoint.checkpoint(self.layer4, x)
            return x

    merlin = Merlin()
    resnet = torchvision.models.resnet152(pretrained=True)
    merlin.model.encode_image.i3_resnet = I3ResnetEncoder(
        copy.deepcopy(resnet), class_nb=1692, conv_class=True
    )
    merlin.model.load_state_dict(
        torch.load(os.path.join(merlin.local_dir, merlin.checkpoint_name))
    )
    encoder = merlin.model.encode_image.i3_resnet.to(args.device)
    # print(emb_model)
    # assert False
    emb_model = lambda x: encoder.encode(x)

else:
    raise 'Unknown model'
#%%

# def crop_pad_matrix(mat, size=256):
#     mat = torch.nn.functional.interpolate(
#         torch.from_numpy(mat[np.newaxis, np.newaxis, :, :]), size=(size, size),
#         mode='trilinear'
#     ).squeeze().numpy()
#     return mat

npzcache = dict()
npzblob = None
if '.npz' in args.dataroot:
    npzblob = np.load(args.dataroot)
    for split in ['train', 'test', 'val']:
        npzcache[split] = dict()
        fs = [f for f in fbatch if split in f]
        fixs = [int(f.split('_')[1]) for f in fs]
        if len(fixs) == 0: continue
        fimin, fimax = min(fixs), max(fixs)
        print(split, fimin, fimax)
        batch = npzblob[f'{split}_images'][fimin:fimax+1]
        for fi in range(fimin, fimax+1):
            npzcache[split][fi] = batch[fi-fimin]


projector = None
if args.k:
    projector = np.load(f'../artifacts/proj_normal_d{latent_size}_k{args.k}_run1.npy').astype(np.float32)

pbar = tqdm(fbatch)
for fname in pbar:
    # NOTE: sample ids may be alphanumeric
    pid = fname

    if '.npz' in args.dataroot:
        npz_split, npz_index = fname.split('_')
        npz_index = int(npz_index)
        # vol = npzblob[f'{npz_split}_images'][npz_index].astype(float)
        vol = npzcache[npz_split][npz_index].astype(float)
        vol /= 256

        if len(vol.shape) < 3:
            vol = vol[:, :, np.newaxis]

        minval = -1
        maxval = -1

        vol = torch.from_numpy(vol.astype(np.float32)).to(args.device)
        vol = torch.nn.functional.interpolate(
            vol[np.newaxis, np.newaxis, :, :, :],
            size=(input_volume_size,)*3,
            mode='trilinear'
        )
        # if args.encoder in ['Merlin']:
        #     vol = vol.repeat(1, 3, 1, 1, 1)
    else:
        raise 'Not implemented'

    with torch.no_grad():
        embs = emb_model(vol)

    # NOTE: to handle some encoders that output multi-scale latents
    if args.encoder == 'Merlin':
        embs = [embs[0]]
    if type(embs) != list:
        embs = [embs]

    # print(emb_model)
    # print(embs.shape)
    # # print(embs[0][0].shape, embs[0][1].shape)
    # assert False
    embs = [e.squeeze() for e in embs]

    ls = []
    for emb_scale in embs:
        if args.k is not None:
            emb_scale = torch.einsum('ij,jklm->iklm', torch.from_numpy(projector.T).to(args.device), emb_scale)

        if args.avgpool:
            # avg over all spatial dims
            npatches = emb_scale.shape[-1]
            reduced = emb_scale.sum((-3, -2, -1)) / (npatches**3)
            ls += [reduced.squeeze()]
        elif args.planar:
            # avg over each spatial dim; this will lead to a x3xSize larger embedding
            npatches = emb_scale.shape[-1]
            for dim in range(3):
                reduced = emb_scale.sum(dim+1) / npatches
                ls += [reduced.squeeze().view(-1)]
        else:
            # print(emb_scale.shape)
            ls += [emb_scale.view(-1)]
            # print(emb_scale.shape)
            # raise 'Not implemented'
    # assert False

    embs_agg = torch.cat(ls).detach().cpu().numpy().astype(np.float32)

    pbar.set_postfix(dict(
        pid=pid, sh=vol.shape, minval=minval, maxval=maxval, d=len(embs_agg)
    ))

    embname = None
    if args.planar:
        embname = 'planar'
        if args.k is not None:
            embname += f'_k{args.k}'
    elif args.avgpool:
        embname = 'avgpool'
    else:
        embname = 'base'
    # else:
    #     raise 'Not implemented'


    if not os.path.exists(f'{args.saveto}/{embname}'):
        os.makedirs(f'{args.saveto}/{embname}')

    np.save(f'{args.saveto}/{embname}/{pid}.npy', embs_agg)

#%%