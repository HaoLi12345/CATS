
import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    KeepLargestConnectedComponentd,
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    Orientationd,
    Resized,
    SaveImaged,
    ScaleIntensityd,
    EnsureTyped,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)


import torch

from model import cats

from monai.data import CacheDataset, DataLoader, Dataset
from glob import glob
from monai.inferers import sliding_window_inference
image_dir = ''
torch.cuda.set_device(0)
val_transforms = Compose(
    [
        LoadImaged(keys=['image']),
        EnsureChannelFirstd(keys='image'),
        #AddChanneld(keys=["image"]),
        CropForegroundd(keys="image", source_key="image"),
        #ToTensord(keys=['image']),
    ]
)


image_paths = sorted(glob(image_dir + '/*.nii.gz'))
root_dir = os.getcwd()
data_dicts = [{'image': image_path,
               'image_path': image_path}
              for image_path in zip(image_paths)]

files = data_dicts
dataset = Dataset(data=files, transform=val_transforms)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

# model = UNETR(
#     in_channels=1,
#     out_channels=14,
#     img_size=(96, 96, 96),
#     feature_size=16,
#     hidden_size=768,
#     mlp_dim=3072,
#     num_heads=12,
#     pos_embed="perceptron",
#     norm_name="instance",
#     res_block=True,
#     dropout_rate=0.0,
# ).cuda()
model = cats.cats(dimensions=3, in_channels=1, out_channels=9,
                  features=(32, 32, 64, 128, 256, 32), image_size=(96, 96, 96)).cuda()

model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model_cats.pth")))

post_transforms = Compose([
    EnsureTyped(keys="pred"),
    Activationsd(keys="pred", softmax=True),
    Invertd(
        keys="pred",  # invert the `pred` data field, also support multiple fields
        transform=val_transforms,
        orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
        # then invert `pred` based on this information. we can use same info
        # for multiple fields, also support different orig_keys for different fields
        meta_keys="pred_meta_dict",  # key field to save inverted meta data, every item maps to `keys`
        orig_meta_keys="image_meta_dict",  # get the meta data from `img_meta_dict` field when inverting,
        # for example, may need the `affine` to invert `Spacingd` transform,
        # multiple fields can use the same meta data to invert
        meta_key_postfix="meta_dict",  # if `meta_keys=None`, use "{keys}_{meta_key_postfix}" as the meta key,
        # if `orig_meta_keys=None`, use "{orig_keys}_{meta_key_postfix}",
        # otherwise, no need this arg during inverting
        nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
        # to ensure a smooth output, then execute `AsDiscreted` transform
        to_tensor=True,  # convert to PyTorch Tensor after inverting
    ),
    AsDiscreted(keys="pred", argmax=True, n_classes=14),
    KeepLargestConnectedComponentd(keys='pred', applied_labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14]),
    SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="./out_cats", output_postfix="seg", resample=False),
])
model.eval()
with torch.no_grad():
    for i, data in enumerate(data_loader, 0):
        val_data = data['image'].cuda()
        data['pred'] = sliding_window_inference(val_data, (96, 96, 96), 1, model)
        # pred = sliding_window_inference(val_data, (96, 96, 96), 1, model)
        data = [post_transforms(i) for i in decollate_batch(data)]
        print(1)

