#%%
import numpy as np

fdim = 256
for seed in range(3):
    np.random.seed(seed)

    # for k in [10, 100]:
    for k in [5,10,15,20,25,30]:
        pmat = np.random.randn(fdim, k)
        np.save(f'../artifacts/proj_normal_d{fdim}_k{k}_run{seed+1}.npy', pmat)
# # %%
# for k in [10, 100]:
#     pmat = np.random.rand(256, k)*2-1
#     np.save(f'artifacts/proj_uniform_k{k}.npy', pmat)
# %%
# other kinds of projections?
#  - information-aware quantized
#  - convolutions that are orthogonal to empty voxels