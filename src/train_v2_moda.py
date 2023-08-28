import os
import numpy as np
from tqdm import tqdm
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    LoadImaged,
    Spacingd,
    ToTensord,
    ScaleIntensityd
)
from monai.metrics import DiceMetric
import torch
from src.model import cats_v2
from monai.data import (
    DataLoader,
    decollate_batch,
    Dataset
)
from glob import glob


import warnings
warnings.simplefilter('ignore')
import logging
logging.disable(logging.WARNING)
def get_train_loader(image_dir, label_dir):
    image_paths = sorted(glob(image_dir + '/*.nii.gz'))
    label_paths = sorted(glob(label_dir + '/*.nii.gz'))

    data_dicts = [{'image': image_path, 'label': label_path,
                   'image_path': image_path, 'label_path': label_path}
                  for image_path, label_path in zip(image_paths, label_paths)]

    files = data_dicts

    Transform = Compose([LoadImaged(keys=['image', 'label']), AddChanneld(keys=['image', 'label']),
                         ScaleIntensityd(keys=['image']),
                         Spacingd(
                             keys=["image", "label"],
                             pixdim=(0.754, 0.4473, 1.13),
                             mode=("bilinear", "nearest"),
                         ),
                         # RandCropByPosNegLabeld(
                         #     keys=["image", "label"],
                         #     label_key="label",
                         #     spatial_size=(64, 64, 64),
                         #     pos=1,
                         #     neg=1,
                         #     num_samples=4,
                         #     image_key="image",
                         #     image_threshold=0,
                         # ),

                         ToTensord(keys=['image', 'label']),
                        ])

    dataset = Dataset(data=files, transform=Transform)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

    case_num = 0
    img_name = os.path.split(dataset[case_num]["image_meta_dict"]["filename_or_obj"])[1]
    img = dataset[case_num]["image"]
    label = dataset[case_num]["label"]
    img_shape = img.shape
    label_shape = label.shape
    print(f"image shape: {img_shape}, label shape: {label_shape}")
    return data_loader


def get_validation_loader(image_dir, label_dir):
    image_paths = sorted(glob(image_dir + '/*.nii.gz'))
    label_paths = sorted(glob(label_dir + '/*.nii.gz'))


    data_dicts = [{'image': image_path, 'label': label_path,
                   'image_path': image_path, 'label_path': label_path}
                  for image_path, label_path in zip(image_paths, label_paths)]

    files = data_dicts

    transform_validation = Compose([LoadImaged(keys=['image', 'label']),
                                    AddChanneld(keys=['image', 'label']),
                                    ScaleIntensityd(keys=['image']),
                                    Spacingd(
                                        keys=["image", "label"],
                                        pixdim=(0.754, 0.4473, 1.13),
                                        mode=("bilinear", "nearest"),
                                    ),
                                    ToTensord(keys=['image', 'label'])
                                   ])

    dataset = Dataset(data=files, transform=transform_validation)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)



    return data_loader


root_dir = os.getcwd()
print(root_dir)


x, y, z = 128,128,64
num_samples = 2
image_dir = ''
label_dir = ''

validation_image_dir = ''
validation_label_dir = ''
train_loader = get_train_loader(image_dir, label_dir)
validation_loader = get_validation_loader(validation_image_dir, validation_label_dir)


model = cats_v2.cats_v2(img_size=(x, y, z), in_channels=1, out_channels=3, feature_size=48,
                        features=(48, 48, 96, 192, 384, 768, 48), 
                        ).cuda()

torch.backends.cudnn.benchmark = True
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
                    model.state_dict(), os.path.join(root_dir, "best_metric_model_moda_swin_cats_full.pth")
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
post_label = AsDiscrete(to_onehot=3)
post_pred = AsDiscrete(argmax=True, to_onehot=3)
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
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model_moda_swin_cats_full.pth")))

print(
    f"train completed, best_metric: {dice_val_best:.4f} "
    f"at iteration: {global_step_best}"
)
