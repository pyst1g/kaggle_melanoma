#!/usr/bin/env python
# coding: utf-8

# In[1]:


import albumentations as A
import cv2
import torch
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2


# In[2]:


def get_train_transforms(input_res):
    return A.Compose([
#             A.JpegCompression(p=0.5),
            A.Rotate(limit=45, p=1.0),
            A.OneOf([
                A.OpticalDistortion(),
                A.GridDistortion(),
#                 A.IAAPiecewiseAffine(),
            ]),
            A.RandomSizedCrop(min_max_height=(int(input_res*0.7), input_res),
                              height=int(input_res*0.85), width=int(input_res*0.85), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.GaussianBlur(p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(),   
                A.HueSaturationValue(),
            ]),
            A.Cutout(num_holes=8, max_h_size=int(input_res*0.85)//8, max_w_size=int(input_res*0.85)//8, fill_value=0, p=0.3),
#             A.Normalize(),
            ToTensorV2(),
        ], p=1.0)

def get_valid_transforms(input_res):
    return A.Compose([
            A.CenterCrop(height=int(input_res*0.85), width=int(input_res*0.85), p=1.0),
#             A.Normalize(),
            ToTensorV2(),
        ], p=1.0)

def get_tta_transforms(input_res):
    return get_train_transforms(input_res)


# In[3]:


class ImageDataset(Dataset):
    def __init__(self, df, labels=None, transforms=None, feature_cols=None):
        super().__init__()
#         self.path = path
        self.df = df
        self.image_ids = df.index.values
        self.data_paths = df.data_path.values
        self.labels = labels
        self.transforms = transforms
        self.feature_cols = feature_cols

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        data_path = self.data_paths[idx]
        image = cv2.imread(f'{data_path}/{image_id}.jpg', cv2.IMREAD_COLOR)
#         print(f'{data_path}/{image_id}.jpg')
        if self.transforms:
            sample = self.transforms(image=image)
            image  = sample['image']
            image = image.float()
            

        label = self.labels[idx] if self.labels is not None else 0.5

        if self.feature_cols is None:
            return image, label
        else: 
            # imageとtable合わせて学習する場合
            x = self.df.loc[image_id, self.feature_cols]
            x = torch.tensor(x).float()
            return image, x, label
            

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def get_labels(self):
        return list(self.labels)

