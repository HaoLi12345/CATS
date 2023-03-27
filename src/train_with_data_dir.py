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
    AddChanneld,
    Compose,
    ScaleIntensityd,
    ToTensord,
    LoadImaged,
    RandSpatialCropd,
    RandAdjustContrastd,
    CropForegroundd,
    RandZoomd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandBiasFieldd,
    RandShiftIntensityd
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    Dataset
)


import torch

from model import cats
from monai.networks.nets import UNETR
from glob import glob
def get_train_loader(image_dir, label_dir, x=64, y=64, z=64):
    image_paths = sorted(glob(image_dir + '/*.nii.gz'))
    label_paths = sorted(glob(label_dir + '/*.nii.gz'))
    data_dicts = [{'image': image_path, 'label': label_path,
                   'image_path': image_path, 'label_path': label_path}
                  for image_path, label_path in zip(image_paths, label_paths)]
    files = data_dicts
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),

            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(x, y, z),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            # RandFlipd(
            #     keys=["image", "label"],
            #     spatial_axis=[0],
            #     prob=0.10,
            # ),
            # RandFlipd(
            #     keys=["image", "label"],
            #     spatial_axis=[1],
            #     prob=0.10,
            # ),
            # RandFlipd(
            #     keys=["image", "label"],
            #     spatial_axis=[2],
            #     prob=0.10,
            # ),
            # RandRotate90d(
            #     keys=["image", "label"],
            #     prob=0.10,
            #     max_k=3,
            # ),
            # RandShiftIntensityd(
            #     keys=["image"],
            #     offsets=0.10,
            #     prob=0.50,
            # ),
            ToTensord(keys=["image", "label"]),
        ]
    )

    dataset = Dataset(data=files, transform=train_transforms)
#     dataset = CacheDataset(
#     data=files,
#     transform=train_transforms,
#     cache_num=24,
#     cache_rate=1.0,
#     num_workers=8,
# )
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

    return data_loader


def get_validation_loader(image_dir, label_dir):
    image_paths = sorted(glob(image_dir + '/*.nii.gz'))
    label_paths = sorted(glob(label_dir + '/*.nii.gz'))
    data_dicts = [{'image': image_path, 'label': label_path,
                   'image_path': image_path, 'label_path': label_path}
                  for image_path, label_path in zip(image_paths, label_paths)]
    files = data_dicts
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ToTensord(keys=["image", "label"]),
        ]
    )
    dataset = Dataset(data=files, transform=val_transforms)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    return data_loader
root_dir = os.getcwd()
print(root_dir)
x, y, z = 96, 96, 96

torch.cuda.set_device(0)
image_dir = 
label_dir = 
train_loader = get_train_loader(image_dir, label_dir, x=x, y=y, z=z)

vali_image_dir = 
vali_label_dir = 
validation_loader = get_validation_loader(vali_image_dir, vali_label_dir)


model = cats.cats(dimensions=3, in_channels=1, out_channels=9,
                  features=(32, 32, 64, 128, 256, 32), image_size=(x, y, z)).cuda()
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

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
# torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)


def validation(epoch_iterator_val):
    model.eval()
    dice_vals = list()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, (x, y, z), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            dice = dice_metric.aggregate().item()
            dice_vals.append(dice)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice)
            )

        dice_metric.reset()
    mean_dice_val = np.mean(dice_vals)
    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
        )
        if (
            global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                validation_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(
                    model.state_dict(), os.path.join(root_dir, "best_metric_model.pth")
                )
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best


max_iterations = 25000
eval_num = 500
post_label = AsDiscrete(to_onehot=9, n_classes=9)
post_pred = AsDiscrete(argmax=True, to_onehot=9, n_classes=9)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(
        global_step, train_loader, dice_val_best, global_step_best
    )
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))

print(
    f"train completed, best_metric: {dice_val_best:.4f} "
    f"at iteration: {global_step_best}"
)
