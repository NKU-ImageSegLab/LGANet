import os
import public.transforms.two as my_transforms
from .image_loader import ImageLoader


class ISICDataset(ImageLoader):
    """
    load ISIC 2018 dataset
    """

    def __init__(self, opt, mode):
        """
        initialize ISIC 2018 dataset
        :param opt: params dict
        :param mode: train/valid
        """
        if mode == "train":
            trans = my_transforms.Compose([
                my_transforms.RandomResizedCrop(tuple(opt["resize_shape"]), scale=(0.4, 1.0), ratio=(3. / 4., 4. / 3.),
                                                interpolation='BILINEAR'),
                my_transforms.ColorJitter(brightness=opt["color_jitter"], contrast=opt["color_jitter"],
                                          saturation=opt["color_jitter"], hue=0),
                my_transforms.RandomGaussianNoise(p=opt["augmentation_p"]),
                my_transforms.RandomHorizontalFlip(p=opt["augmentation_p"]),
                my_transforms.RandomVerticalFlip(p=opt["augmentation_p"]),
                my_transforms.RandomRotation(opt["random_rotation_angle"]),
                my_transforms.Cutout(p=opt["augmentation_p"], value=(0, 0)),
                my_transforms.ToTensor(),
                my_transforms.Normalize(mean=opt["normalize_means"], std=opt["normalize_stds"])
            ])
        else:
            trans = my_transforms.Compose([
                my_transforms.Resize(opt["resize_shape"]),
                my_transforms.ToTensor(),
                my_transforms.Normalize(mean=opt["normalize_means"], std=opt["normalize_stds"])
            ])

        base_path = os.path.join(opt["dataset_path"], "train" if mode == "train" else "val")

        super(ISICDataset, self).__init__(
            origin_image_path=os.path.join(base_path, "images"),
            gt_image_path=os.path.join(base_path,"masks"),
            mode=mode,
            transforms=trans,
            support_types=['jpg', 'png', 'jpeg', 'bmp', 'tif', 'tiff', 'JPG', 'PNG', 'JPEG', 'BMP', 'TIF', 'TIFF'],
            gt_format=opt["gt_format"]
        )
