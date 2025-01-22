#%%
import numpy as np
#%%
droot = '/u/project/sgss/UKBB/medmnist'
# dset = 'amish'
dset = 'bimcvr'
# dset = 'lidc'
# dset = 'lungx'
# dset = 'ccccii'
# %%
blob = np.load(f'{droot}/{dset}mnist3d_64.npz')
# %%
ids = []
for ph in ['train', 'val', 'test']:

    labs = blob[f'{ph}_labels']
    cats = np.unique(labs[:, 0])
    if ph == 'train': print(cats)
    prevs = [(labs == c).sum() / len(labs) for c in cats]
    print(ph, len(labs), ['%.4f' % f for f in prevs])
    ids += [f'{ph}_{i}' for i in range(len(labs))]
# %%
np.savetxt(f'saved/manifests/{dset}mnist3d.txt', ids, fmt='%s')
# %%
# ytrain = blob['train_labels'][:]
# ytrain[ytrain == 3] = 0
# #%%
# savedict = { k: v for k, v in blob.items()}
# savedict['train_labels'] = ytrain
# #%%
# # %%
# np.savez_compressed(f'{droot}/{dset}fixmnist3d_64.npz', **savedict)
# %%
# import matplotlib.pyplot as plt

# vol = blob['train_images'][0]
# vol.shape

# plt.figure()
# plt.imshow(vol[:, :, 32])
# plt.show()
# vol.min(), vol.max()
# %%
