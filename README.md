# Raptor benchmarks

Run Raptor and related methods on MedMNIST formatted datasets.

## MedMNIST data format

MedMNIST datasets are in npz format with keys 'train_images', 'train_labels', etc...
for train, val, and test. The 'train_images' entry should contain volumes in N x H x W x D.
The labels are assumed to be classification with mutually exclusive classes in the shape N x 1
where each entry ranges from 0...C .

The datasets are expected to be in `../medmnist`. If it is at a different path, a symlink can be created at this location for convenience.

## Extended MedMNIST data format

We also store multi-label classification and regression tasks using a similar format as described.
For multi-label, we define a one-hot indicator for the '*_labels' entry in the shape N x C.
For regression, we store the quantitative labels (can be multi dimensional) also in the shape N x C.

We allow volumes to be stored outside the npz due to performance/storage issues. If this is the case, partial paths to the volumes (string) should be stored in '*_images' instead of the actual volumes.
The training scripts should handle some `root_path` option such that the volumes can be located at different drives on different machines.

## Main training scripts

* `fit.py`: Logistic regression on precomputed embeddings (Raptor, VoCo)
* `fit_e2e.py`: End to end training for pretrained models (PCTNet, SuPreM, ResNet, Merlin)
* `fit_mae.py`: MAE training and classification training for standard ViT (MAE)

