#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Author   : Guo Qingqing
# @Date     : 2022/10/12 下午8:15
# @Software : PyCharm

import torch
import torch.nn.functional as F
import numpy as np
import os, argparse

from utils.image_loader.ISICDataset import ISICDataset
from utils.loader import *
import yaml
from PIL import Image
import imageio
from tqdm import tqdm
from models.model import LGANet
from utils.metrics_info.metrics import get_binary_metrics, MetricsResult

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/config_skin_isic2016.yml', help="config file path")
    parser.add_argument("--dataset_path", type=str, default=None, help="dataset name")
    args = parser.parse_args()
    return args


## Hyper parameters
args = parse_args()
with open(args.config, "r") as yaml_file:
    # 使用PyYAML加载YAML数据
    config = yaml.safe_load(yaml_file)
config["dataset_path"] = args.dataset_path if args.dataset_path is not None else config["dataset_path"]
number_classes = int(config['number_classes'])
input_channels = 3
best_val_loss = np.inf
patience = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_dataset = ISICDataset(config, mode='val')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# In[3]:
model_name = 'LGANet'

save_result_path = config['save_result']
if not os.path.exists(save_result_path):
    os.makedirs(save_result_path)

save_result_pred_path = config['save_result']
if not os.path.exists(save_result_pred_path):
    os.makedirs(save_result_pred_path)
save_best_loss_model_path = os.path.join(config['weight_path'], 'best_loss_weight_path', config['saved_model'])
Net = LGANet()
Net = Net.to(device)
Net.load_state_dict(
    torch.load(save_best_loss_model_path.__str__(), map_location='cpu')[
        'model_weights'])

with torch.no_grad():
    print('val_mode')
    val_loss = 0
    Net.eval()
    vl_metrics = get_binary_metrics()
    for itter, (img, msk, image_name) in tqdm(
            iterable=enumerate(test_loader),
            desc=f"{config['dataset_name']} Validation",
            total=len(test_loader)
    ):
        img = img.to(device, dtype=torch.float)
        msk = msk.to(device, dtype=torch.float32)
        msk_pred, _, _, _ = Net(img)
        vl_metrics.update(msk_pred, msk.int())
        msk_pred = torch.sigmoid(msk_pred)

        msk_pred_output = msk_pred.cpu().detach().numpy()[0, 0] > 0.5
        msk_pred_output = (msk_pred_output * 255).astype(np.uint8)
        # name = 'test_'+str(index)+'.png'
        # print(name)
        total_length = len(image_name)
        for i in range(len(image_name)):
            name = image_name[i] + '.png'
            if total_length == 1:
                predict_image = msk_pred_output
            else:
                predict_image = msk_pred_output[i]
            imageio.imwrite(os.path.join(save_result_path, name), predict_image)
            # pred = msk_pred[i].cpu().detach().numpy()[0, 0]
            # pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            # pred = (pred * 255).astype(np.uint8)
            # print(pred.shape)
            # imageio.imwrite(os.path.join(save_result_pred_path, name), pred)
        # 概率图
    vl_metrics_result = MetricsResult(vl_metrics.compute())
    params, flops = vl_metrics_result.cal_params_flops(Net, 256)
    vl_metrics_result.to_result_csv(
        os.path.join(save_result_path, 'result.csv'),
        model_name,
        flops=flops,
        params=params
    )

if __name__ == '__main__':
    pass