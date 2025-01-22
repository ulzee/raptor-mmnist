
import numpy as np
from torch.utils.data import Dataset
import os
from glob import glob

class MM3dVolDataset(Dataset):
    def __init__(self, split, datapath, bulkpath=None, stats=None):
        self.split = split
        self.datapath = datapath
        self.stats = stats
        self.bulkpath = bulkpath

        self.vols = np.load(f'{datapath}')[f'{split}_images']
        self.labs = np.load(f'{datapath}')[f'{split}_labels'][:]

    def __len__(self):
        return len(self.labs)

    def __getitem__(self, index):
        vol = self.vols[index]
        if isinstance(vol, np.str_):
            vol = npy_npz_priority_load(self.bulkpath + '/' + vol)
        lab = self.labs[index]
        return vol, lab

def npy_npz_priority_load(fname):
    froot = fname.replace('.npz', '').replace('.npy', '')
    npy = froot + '.npy'
    npz = froot + '.npz'
    t = None
    if os.path.exists(npy):
        t =  np.load(npy)
    else:
        t =  np.load(npz)['arr_0']

    try:
        assert not np.isnan(np.sum(t))
    except:
        print(fname)
        assert False

    return t

class MM3DEmbSimDataset(Dataset):
    def __init__(self, split, task, labels_dir, embs_dir, split_ids, stats=None, args=dict()):

        # self.root_dir = root_dir
        self.embs_dir = embs_dir
        self.split = split
        self.stats = stats
        self.split_ids = split_ids
        self.task = task
        self.args = args

        self.labels = np.load(f'{labels_dir}/simulated/{task}/{split}_labels.npy')

    def __len__(self):
        return len(self.split_ids)

    def __getitem__(self, index):

        pid = self.split_ids[index]
        pix = int(pid.split('_')[1])

        evec = npy_npz_priority_load(f'{self.embs_dir}/{pid}.npz')
        evec /= 256 # NOTE: this is the standard input pad size before embedding

        lbl = self.labels[pix]
        # if self.args.task == 'location':
        #     # turn the loc into a all-or-nothing binary classification
        #     cls_idx = [4, 8, 16, 32, 64, 128].index(int(self.args.etc))
        #     lbl = [[0, 1][int(self.labels[pix] == cls_idx)]]

        return evec, lbl

class MM3DEmbDataset(Dataset):
    def __init__(self, split, task, labels_dir, embs_dir, split_ids, stats=None, args=dict()):

        # self.root_dir = root_dir
        self.embs_dir = embs_dir
        self.split = split
        self.stats = stats
        self.split_ids = split_ids

        blob = np.load(f'{labels_dir}/{task}_64.npz')
        self.blob = blob
        self.labels = blob[f'{split}_labels'] # NOTE: make sure to index correctly if using subsamples

    def __len__(self):
        return len(self.blob[f'{self.split}_labels'])

    def __getitem__(self, index):

        pid = self.split_ids[index]
        pix = int(pid.split('_')[1])

        evec = npy_npz_priority_load(f'{self.embs_dir}/{pid}.npz')
        evec /= 256 # NOTE: this is the standard input pad size before embedding
        return evec, self.labels[pix]

class MM3DRetrievalDataset(MM3DEmbDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        doc_emb_files = [f.replace('npys/', 'docs/') for f in self.blob[f'{self.split}_images']]
        # doc_emb_files = [(i, f'/u/scratch/u/ulzee/{f}') for i, f in enumerate(doc_emb_files)]
        doc_emb_files = [(i, f'/u/scratch/u/ulzee/{f}') for i, f in enumerate(doc_emb_files)]

        # subsample val to train faster
        doc_emb_files = [(i, f) for i, f in doc_emb_files if os.path.exists(f)]
        # doc_emb_files = [(i, f) for i, f in doc_emb_files if os.path.exists(f) \
        #                  and (self.split != 'val' or i % 50 == 0)]
                        #  and i % 50 == 0]
        # doc_emb_files = doc_emb_files[:1]
        # doc_emb_files = [(i, f) for i, f in doc_emb_files if os.path.exists(f)]

        doc_emb_files_inds, self.doc_emb_files = map(list, zip(*doc_emb_files))

        # filter out doc embs that don't exist for any reason
        filtered_blob = {}
        filtered_blob[f'{self.split}_images'] = self.blob[f'{self.split}_images'][doc_emb_files_inds]
        filtered_blob[f'{self.split}_labels'] = self.blob[f'{self.split}_labels'][doc_emb_files_inds]
        self.blob = filtered_blob
        self.split_ids = [self.split_ids[i] for i in doc_emb_files_inds] # NOTE: superclass uses split_ids to get vols, be careful
        print(f'Found {len(doc_emb_files_inds)} matching documents')

    def __getitem__(self, index):
        evec, _ = super().__getitem__(index)

        # FIXME: pass local path
        docemb = np.load(self.doc_emb_files[index])

        return evec, docemb

class MM3DEmbSeqDataset(MM3DEmbDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.k = int(kwargs['embs_dir'].split('_k')[1].split('_')[0])

    def __getitem__(self, index):
        evec, lab = super().__getitem__(index)
        return evec.reshape(3, self.k, -1), lab