

# LGANet: Local-Global Augmentation Network for Skin Lesion Segmentation
This repo is the official implementation  for the paper: **"LGANet: Local-Global Augmentation Network for Skin Lesion Segmentation"** at ISBI 2023.

This paper proposes a novel framework, LGANet, for skin lesion segmentation. Particularly, two module, LFM and GAM are constructed. LFM aims at learning local inter-pixel correlations to augment local detailed information around boundary regions, while GAM aims at learning global context at a finer level to augment global information.
## Architecture
![Network](https://img-blog.csdnimg.cn/bf41c11f82ec4cd382d3dd916829de98.png#pic_center)Fig.2. The structure of the proposed LGANet. LFM and GAM are integrated into the Transformer encoder based framework to learn local detailed information around boundary and augment global context respectively, where dense concatenations are used for final pixel-level prediction.

## System Requirement
We need to run the code on linux.

## Requirements Installation
```bash
sh setup.sh
```

## Datasets

 - The ISIC 2018  and ISIC 2016 dataset can be acquired from [the official site](https://challenge.isic-archive.com/data/).
 - ~~Run Prepare_data.py for data preperation.~~No needed to run Prepare_data.py. You can just select a config file by using the `--config` option in the train.py file and if you provide `--dataset_path` option, the code prepare the data for training.


## Pretrained model
You could download the pretrained model from [here](https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV).  Please put it in the " **./pretrained**" folder for initialization.
## Training
```shell
python train.py --config configs/config_skin_isic2016.yml --dataset_path /path/to/your/dataset
``` 
## Testing
```shell
python test.py --config configs/config_skin_isic2016.yml --dataset_path /path/to/your/dataset
``` 
## Evaluation
Note: evaluate is not supported now. We will update it soon.
## References
Some of the codes in this repo are borrowed from:
 - [TMUNet](https://github.com/rezazad68/TMUnet)     
 - [PVT](https://github.com/whai362/PVT)


## Citation
Please cite the following paper if you think this project is useful for your work. Thanks.

@inproceedings{

GuoFWZL2023LGANet,

author = { Guo, Qingqing and Fang, Xianyong and Wang, Linbo and Zhang, Enming and Liu, Zhengyi},

booktitle = {Proceedings of the 20th IEEE International Symposium on Biomedical Imaging - ISBI 2023},

title = {{LGANet: Local-global augmentation network for skin lesion segmentation}},

address = {Cartagena, Colombia},

year = {2023}

}

