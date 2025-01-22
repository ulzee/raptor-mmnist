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

## Raptor scripts overview

The scripts relevant to raptor are:

* `embed.py`: Uses Raptor to embed volumes.
* `fit.py`: Logistic regression on precomputed embeddings (Raptor, VoCo)

## Training scripts overview

* `fit.py`: Logistic regression on precomputed embeddings (Raptor, VoCo)
* `fit_e2e.py`: End to end training for pretrained models (PCTNet, SuPreM, ResNet, Merlin)
* `fit_mae.py`: MAE training and classification training for standard ViT (MAE)

### Adding new methods to test in our pipeline:

Generally it is recommended to add new methods to compare with at the top of `fit_mae.py` so that we can conveniently use one script to run all experiments.
For example, PCT-Net was added in the following way:

Import the model definition from a local path and define default options for this model:
```python
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
```

Wrap the model to do classification:
```python
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
```

Load pretrained weights shared by the authors of the model:
```python
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
```

