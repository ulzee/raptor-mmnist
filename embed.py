#%%
import os, sys
import numpy as np
from tqdm import tqdm
import torch
torch.set_num_threads(4)
import shutil
import nibabel as nib
import zipfile
import argparse
from time import time


from datasets import npy_npz_priority_load
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
#%%
class args:
    dataroot = '/u/project/sgss/UKBB/medmnist/fracturemnist3d_64.npz'
    # dataroot = '/u/project/sgss/UKBB/medmnist/simulated/location'
    # encoder = 'DINO'
    # dataroot = '/u/project/sgss/UKBB/medmnist/location'
    encoder = 'SAM'
    manifest = 'saved/manifests/missing_SAM_fracture.txt'
    device = 'cuda:0'
    start = 0
    many = 66
    batch_size = 64
    saveto = '/data2/bsplat/data/embs/dec19_simulated_location_DINO_nodule'
    # all_slices = True
    k = '100'
    planar = True
    avgpool = False
    planes = 'A,C,S'
    save_sam = False
#%%
parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--dataroot', type=str, default='/u/project/sgss/UKBB/imaging/bulk/20253', help='Data root directory')
parser.add_argument('--encoder', type=str, default='SAM', help='Encoder type')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
parser.add_argument('--manifest', type=str, required=True, help='Manifest file path')
parser.add_argument('--start', type=int, required=True, help='Start index')
parser.add_argument('--many', type=int, required=True, help='Number of files to process')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--saveto', type=str, required=True, help='Save directory')
parser.add_argument('--k', type=str, default=None)
# parser.add_argument('--all_slices', default=True)
parser.add_argument('--no_flatviews', default=False, action='store_true')
parser.add_argument('--planar', default=False, action='store_true')
parser.add_argument('--planes', default='A,C,S')
parser.add_argument('--avgpool', default=False, action='store_true')
parser.add_argument('--save_sam', action='store_true', default=False)

args = parser.parse_args()
#%%
args.planes = args.planes.split(',')
for p in args.planes:
    assert p in 'ACS'
if args.k is not None:
    args.k = [int(i) for i in args.k.split(',')]
assert args.avgpool or args.planar
if args.avgpool:
    assert args.k is None # there is no need for Ks

with open(args.manifest) as fl:
    fls = [ln.strip() for ln in fl if ln]
fbatch = fls[args.start:args.start+args.many]
fbatch
#%%
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
elif 'simulated' in args.dataroot:
    pass
    # for sample_id in tqdm(fbatch):
    #     samp_split, samp_ix = sample_id.split('_')
    #     samp_ix = int(samp_ix)
    #     if samp_split not in npzcache: npzcache[samp_split] = dict()
    #     npzcache[samp_split][samp_ix] = np.load(f'{args.dataroot}_{sample_id}.npz')['arr_0'][0]
else:
    raise 'Not implemented'
#%%
latent_size = dict(
    SAM=256,
    MedSAM=256,
    CLIP=1024,
    DINO=1024,
)[args.encoder]

if args.encoder in ['MedSAM', 'SAM']:
    sys.path.append('../etc/MedSAM')
    from segment_anything import sam_model_registry
    sam_checkpoints = dict(
        MedSAM='../etc/MedSAM/work_dir/MedSAM/medsam_vit_b.pth',
        SAM='../etc/MedSAM/work_dir/MedSAM/sam_vit_b_01ec64.pth',
    )
    medsam_model = sam_model_registry["vit_b"](checkpoint=sam_checkpoints[args.encoder])
    medsam_model = medsam_model.to(args.device)
    medsam_model.eval()
elif args.encoder in ['DINO', 'CLIP']:
    # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    # image = Image.open(requests.get(url, stream=True).raw)

    model_url = dict(
        DINO='facebook/dinov2-large',
        CLIP='openai/clip-vit-large-patch14'
    )[args.encoder]

    processor = AutoImageProcessor.from_pretrained(model_url)
    medsam_model = AutoModel.from_pretrained(model_url).to(args.device)
    medsam_model.eval()

    if args.encoder == 'CLIP':
        medsam_model = medsam_model.vision_model.eval()

    # inputs = processor(images=image, return_tensors="pt")
    # outputs = model(**inputs)
    # last_hidden_states = outputs.last_hidden_state
else:
    raise 'Unknown model'
#%%
# from torchvision.transforms import Resize
def crop_pad_matrix(mat, size=224):
    if mat.shape[-1] == size:
        # nothing to do
        return mat

    mat = torch.nn.functional.interpolate(
        torch.from_numpy(mat[np.newaxis, np.newaxis, :, :]), size=(size, size),
        mode='bicubic'
    ).squeeze().numpy()
    return mat

projfiles = []
pbar = tqdm(fbatch)
for fname in pbar:
    # pid = fname.split('/')[-1].split('_')[0]

    # FIXME:
    # pid = int(fname.split('/')[-1].split('_')[1])
    pid = fname

    if '.npz' in args.dataroot or 'simulated' in args.dataroot:
        npz_split, npz_index = fname.split('_')
        npz_index = int(npz_index)
        # vol = npzblob[f'{npz_split}_images'][npz_index].astype(float)
        if 'simulated' not in args.dataroot:
            vol = npzcache[npz_split][npz_index].astype(float)
        else:
            if 'location' in args.dataroot:
                vol = npy_npz_priority_load(f'{args.dataroot}_{fname}.npz')[0].astype(float)
            elif 'shape' in args.dataroot:
                vol = npy_npz_priority_load(f'{args.dataroot}_{fname}.npz')[0].astype(float)
            else:
                raise 'Not implemented'
        vol /= 256

        if len(vol.shape) < 3:
            vol = vol[:, :, np.newaxis]

        minval = -1
        maxval = -1
    else:
        zipname = f'{args.dataroot}/{fname}'


        temp_folder = f'temp_{args.saveto}/{pid}'
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
        os.makedirs(temp_folder, exist_ok=True)
        with zipfile.ZipFile(zipname, 'r') as zip_ref:
            if 'T2_FLAIR/T2_FLAIR_brain_to_MNI.nii.gz' in zip_ref.namelist():
                zip_ref.extract('T2_FLAIR/T2_FLAIR_brain_to_MNI.nii.gz', path=temp_folder)
                file_path = f'{temp_folder}/T2_FLAIR/T2_FLAIR_brain_to_MNI.nii.gz'
            else:
                continue

        shutil.rmtree(temp_folder)
        vol = nib.load(file_path).get_fdata()

        minval, maxval = vol.min(), vol.max()
        # pbar.set_postfix(dict(
        #     pid=pid, min=minval, max=maxval, sh=vol.shape
        # ))

        hclip = 1024+256
        vol[vol < 0] = 0
        vol[vol > hclip] = hclip
        vol = vol.astype(float)
        vol /= hclip

    # print(vol.shape)

    # collect slices (in axes order)
    slices_byxyz = []

    ntot = []
    if 'A' in args.planes:
        slices = []
        for i in range(0, vol.shape[0]-0):
            if vol[i].shape[0] == 1 or vol[i].shape[1] == 1: continue
            slices += [vol[i]]
        if not args.no_flatviews:
            slices += [vol.mean(0)]
        nx = len(slices)
        ntot += [nx]
        slices_byxyz += slices

    if 'C' in args.planes:
        slices = []
        for i in range(0, vol.shape[1]-0):
            if vol[:, i].shape[0] == 1 or vol[:, i].shape[1] == 1: continue
            slices += [vol[:, i]]
        if not args.no_flatviews:
            slices += [vol.mean(1)]
        ny = len(slices)
        ntot += [ny]
        slices_byxyz += slices

    if 'S' in args.planes:
        slices = []
        for i in range(0, vol.shape[2]-0):
            if vol[:, :, i].shape[0] == 1 or vol[:, :, i].shape[1] == 1: continue
            slices += [vol[:, :, i]]
        if not args.no_flatviews:
            slices += [vol.mean(2)]
        nz = len(slices)
        ntot += [nz]
        slices_byxyz += slices

    assert len(slices_byxyz)

    t0 = time()
    imgs = [crop_pad_matrix(img) for img in slices_byxyz]
    t_crop = time() - t0
    embs = []
    t0 = time()
    for i in range(0, len(imgs), args.batch_size):
        if args.encoder in ['MedSAM', 'SAM']:
            imgbatch = np.array(imgs[i:i+args.batch_size]).astype(np.float32)
            imgbatch = torch.from_numpy(imgbatch[:, None]).repeat(1, 3, 1, 1).to(args.device)
            with torch.no_grad():
                embs += [e for e in medsam_model.image_encoder(imgbatch).detach().cpu().numpy()]
        else:
            imgbatch = [Image.fromarray((i*256).astype(np.uint8)) for i in imgs[i:i+args.batch_size]]
            with torch.no_grad():
                inputs = processor(images=imgbatch, return_tensors="pt").to(args.device)
                outputs = medsam_model(**inputs)
                last_hidden_states = outputs.last_hidden_state
                embs += [e.T for e in last_hidden_states.detach().cpu().numpy()]
    t_inf = time() - t0

    if args.save_sam:
        np.savez_compressed(f'{args.saveto}/{pid}.npz', np.array(embs).astype(np.float32))
        continue

    fdim = embs[0].shape[0]
    # for projector in ['../artifacts/proj_normal_k10.npy', '../artifacts/proj_normal_k100.npy', None]:
    # for projector in ['../artifacts/proj_normal_k10.npy', '../artifacts/proj_normal_k100.npy']:

    if len(projfiles) == 0:
        if args.k is not None:
            for kproj in args.k:
                for run in range(1, 3+1):
                    pfile = f'../artifacts/proj_normal_d{fdim}_k{kproj}_run{run}.npy'
                    projfiles += [(pfile, np.load(pfile))]
        else:
            projfiles = [('proj_identity', None)]

    tls = []
    for (projname, projmat) in projfiles:
        if projmat is not None:
            projname = projname.split('/')[-1].split('.')[0]
            if len(args.planes) < 3:
                projname = f'p{"".join(args.planes)}_{projname}'

            # embs: slices (~16) x 256 x 16 x 16
            assert len(projmat) == len(embs[0])
            t0 = time()
            proj_embs = np.stack([projmat.T @ e.reshape(projmat.shape[0], -1) for e in embs])
        else:
            proj_embs = np.array(embs).reshape(len(embs), -1)
        if not os.path.exists(f'{args.saveto}/{projname}'):
            os.makedirs(f'{args.saveto}/{projname}')
        print(projname, time() - t0)

        # proj_embs: slices (16 + 16 + 16) x K x 256
        # proj_embs_sum: slices 3 x K x 16 x 16
        assert np.sum(ntot) == len(proj_embs)

        if args.planar:
            plane_breaks = []
            agg = 0
            for s_count in ntot:
                plane_breaks += [s_count + agg]
                agg += s_count

            byside = [s for s in np.split(proj_embs, plane_breaks, axis=0) if len(s)]
            # assert len(byside) == 3
            byside = [side.sum(0) for side in byside]
            proj_embs_sum = np.concatenate(byside)

            # proj_embs_sum: slices 3K x 256 ~ 7680 for K=10
            proj_embs_sum_flat = proj_embs_sum.reshape(-1)
        elif args.avgpool:
            proj_embs = proj_embs.reshape(len(proj_embs), latent_size, -1)
            proj_embs_sum_flat = proj_embs.mean((0, -1))
        else:
            raise 'Not implemented'
        # tls += ['%.4f' % (time() - t0)]

        # print()
        # np.savez_compressed(f'{args.saveto}/{projname}/{pid}.npz', proj_embs_sum_flat)
        np.save(f'{args.saveto}/{projname}/{pid}.npy', proj_embs_sum_flat.astype(np.float32))

    # t_save = time() - t0

    pbar.set_postfix(dict(
        # pid=pid, sh=vol.shape, minval=minval, maxval=maxval, ns=len(slices_byxyz), d=proj_embs_sum_flat.shape
        pid=pid, sh=vol.shape[0], ns=len(slices_byxyz), d=proj_embs_sum_flat.shape,
        # t=('%.4f' % t_crop, '%.4f' % t_inf, tls),
    ))

    # assert False
#%%