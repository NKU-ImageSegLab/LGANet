#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division

import argparse
import os

from public.image_loader.ISICDataset import ISICDataset
from public.metrics_info.metrics import get_binary_metrics, MetricsResult

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from public.loader import *
import pandas as pd
import glob
import nibabel as nib
import numpy as np
import copy
import yaml
from tqdm import tqdm
import torch.nn.functional as F
from einops import rearrange, repeat
from models.model import LGANet


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/config_skin_isic2016.yml', help="config file path")
    parser.add_argument("--dataset_path", type=str, default=None, help="dataset name")
    parser.add_argument("--tensorboard_dir", type=str, default=None, help="tensorboard directory")
    args = parser.parse_args()
    return args


## Loader
## Hyper parameters
args = parse_args()
with open(args.config, "r") as yaml_file:
    # 使用PyYAML加载YAML数据
    config = yaml.safe_load(yaml_file)
config["dataset_path"] = args.dataset_path if args.dataset_path is not None else config["dataset_path"]

number_classes = int(config['number_classes'])
input_channels = 3
best_val_loss = np.inf
best_Dice = 0.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset = ISICDataset(config, mode='train')
train_loader = DataLoader(train_dataset, batch_size=int(config['batch_size_tr']), shuffle=True)

val_dataset = ISICDataset(config, mode='val')
val_loader = DataLoader(val_dataset, batch_size=int(config['batch_size_va']), shuffle=False)

# In[3]:
model_name = 'LGANet'
Net = LGANet(channel=32, n_classes=number_classes, pretrain_model_path=config['pretrain_model_path'])

Net = Net.to(device)
optimizer = optim.Adam(Net.parameters(), lr=float(config['lr']))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=config['patience'])
criteria = torch.nn.BCELoss()

# criteria_boundary  = torch.nn.BCELoss()
# criteria_region = torch.nn.MSELoss()

need_metrics_in_train = config['need_metrics_in_train']

end_epoch = int(config['epochs'])
if need_metrics_in_train:
    metrics = get_binary_metrics()
for ep in range(end_epoch):
    Net.train()
    epoch_loss = 0
    for itter, (img, msk, _) in tqdm(
            iterable=enumerate(train_loader),
            desc=f"{config['dataset_name']} Training [{ep + 1}/{end_epoch}]",
            total=len(train_loader)
    ):
        img = img.to(device, dtype=torch.float)
        msk = msk.to(device, dtype=torch.float32)

        msk_pool2 = rearrange(msk, 'b c (h n) (w m) -> b c h w (n m)', h=8, w=8)  # patch_size=4
        msk_pool2 = torch.sum(msk_pool2, dim=-1)
        msk_pool2[msk_pool2 == 0] = 0
        msk_pool2[msk_pool2 == 16] = 0
        msk_pool2[msk_pool2 > 0] = 1

        msk_pool3 = rearrange(msk, 'b c (h n) (w m) -> b c h w (n m)', h=4, w=4)  # patch_size=4
        msk_pool3 = torch.sum(msk_pool3, dim=-1)
        msk_pool3[msk_pool3 == 0] = 0
        msk_pool3[msk_pool3 == 16] = 0
        msk_pool3[msk_pool3 > 0] = 1

        msk_pool4 = rearrange(msk, 'b c (h n) (w m) -> b c h w (n m)', h=2, w=2)  # patch_size=4
        msk_pool4 = torch.sum(msk_pool4, dim=-1)
        msk_pool4[msk_pool4 == 0] = 0
        msk_pool4[msk_pool4 == 16] = 0
        msk_pool4[msk_pool4 > 0] = 1

        msk_pred, s2, s3, s4 = Net(img)
        if need_metrics_in_train:
            metrics.update(msk_pred, msk.int())
        # msk_pred = torch.sigmoid(msk_pred)
        loss_seg = structure_loss(msk_pred, msk)
        loss_score2 = criteria(s2, msk_pool2)
        loss_score3 = criteria(s3, msk_pool3)
        loss_score4 = criteria(s4, msk_pool4)

        tloss = 0.7 * loss_seg + 0.1 * loss_score2 + 0.1 * loss_score3 + 0.1 * loss_score4
        optimizer.zero_grad()
        tloss.backward()
        optimizer.step()
        epoch_loss += loss_seg.item()
    metrics_result = MetricsResult(metrics.compute())
    print(metrics_result.to_log('Train', ep, int(config['epochs']), epoch_loss / (itter + 1)))
    ## Validation phase
    if need_metrics_in_train:
        metrics.reset()

    with torch.no_grad():
        print('val_mode')
        val_loss = 0
        Net.eval()
        for itter, (img, msk, _) in tqdm(
            iterable=enumerate(val_loader),
            desc=f"{config['dataset_name']} Validation [{ep + 1}/{end_epoch}]",
            total=len(val_loader)
        ):
            img = img.to(device, dtype=torch.float)
            mask_type = torch.float32
            msk = msk.to(device=device, dtype=mask_type)
            # msk_pred,side_out = Net(img)
            msk_pred, _, _, _ = Net(img)
            if need_metrics_in_train:
                metrics.update(msk_pred, msk.int())
            # msk_pred = torch.sigmoid(msk_pred)
            loss = structure_loss(msk_pred, msk)
            val_loss += loss.item()
        vl_metrics_result = MetricsResult(metrics.compute())
        print(vl_metrics_result.to_log('Val', ep, int(config['epochs']), val_loss / (itter + 1)))
        if need_metrics_in_train:
            metrics.reset()
        mean_val_loss = (val_loss / (itter + 1))
        # Check the performance and save the model
        if (mean_val_loss) < best_val_loss:
            print('New best loss, saving...')
            best_val_loss = copy.deepcopy(mean_val_loss)
            state = copy.deepcopy({'model_weights': Net.state_dict(), 'val_loss': best_val_loss})

            save_best_loss_model_path = config['best_model_path']
            save_best_loss_model_path_dir = os.path.dirname(save_best_loss_model_path)
            if not os.path.exists(save_best_loss_model_path_dir):
                os.makedirs(save_best_loss_model_path_dir)
            torch.save(state, save_best_loss_model_path)

    scheduler.step(mean_val_loss)

print('Trainng phase finished')

if __name__ == '__main__':
    pass
