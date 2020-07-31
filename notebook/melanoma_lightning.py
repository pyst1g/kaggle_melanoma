#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import warnings
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy
import seaborn as sns
import torch
# from packages.lpproj_LPP import LocalityPreservingProjection
from lpproj import LocalityPreservingProjection
from pytorch_lightning.callbacks import EarlyStopping
# from utils import *
from scipy import linalg
from sklearn.metrics import (accuracy_score, adjusted_mutual_info_score,
                             adjusted_rand_score, confusion_matrix, f1_score,
                             plot_confusion_matrix)
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms
from torchvision.datasets import MNIST

sys.path.append('../scripts')

warnings.filterwarnings('ignore')


# #### variable

# In[2]:


fold_number = 0
batch_size = 128


# #### data load 

# In[3]:


DATA_PATH = '../input/jpeg-melanoma-512x512/'
TRAIN_ROOT_PATH = f'{DATA_PATH}/train'
TEST_ROOT_PATH = f'{DATA_PATH}/test'

df_train = pd.read_csv(f'{DATA_PATH}/train.csv', index_col="image_name")
df_train["fold"] = df_train["tfrecord"] // 5
df_train = df_train.drop("tfrecord", axis=1)

df_test = pd.read_csv(f'{DATA_PATH}/test.csv', index_col="image_name")

_ = df_train.groupby('fold').target.hist(alpha=0.4)
df_train.groupby('fold').target.mean().to_frame('ratio').T


# In[4]:


# DATA_PATH = '../input/melanoma-merged-external-data-512x512-jpeg'
# TRAIN_ROOT_PATH = f'{DATA_PATH}/512x512-dataset-melanoma/512x512-dataset-melanoma'
# TEST_ROOT_PATH = f'{DATA_PATH}/512x512-test/512x512-test'

# df_folds = pd.read_csv(f'{DATA_PATH}/folds.csv', index_col='image_id',
#                        usecols=['image_id', 'fold', 'target'], dtype={'fold': np.byte, 'target': np.byte})

# df_test = pd.read_csv(f'../input/siim-isic-melanoma-classification/test.csv', index_col='image_name')

# _ = df_folds.groupby('fold').target.hist(alpha=0.4)
# df_folds.groupby('fold').target.mean().to_frame('ratio').T


# In[5]:


class ImageDataset(Dataset):
    def __init__(self, path, image_ids, labels=None, transforms=None, test=False):
        super().__init__()
        self.path = path
        self.image_ids = image_ids
        self.labels = labels
        self.transforms = transforms
        self.test = test

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image = cv2.imread(f'{self.path}/{image_id}.jpg', cv2.IMREAD_COLOR)

        if self.transforms:
            sample = self.transforms(image=image)
            image  = sample['image']
        
        if self.test:
            return image
        else:
            label = self.labels[idx]# if self.labels is not None else 0.5
            return image, label

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def get_labels(self):
        return list(self.labels)


# In[6]:


resolution = 456
input_res  = 512

def get_train_transforms():
    return A.Compose([
#             A.JpegCompression(p=0.5),
#             A.Rotate(limit=80, p=1.0),
#             A.OneOf([
#                 A.OpticalDistortion(),
#                 A.GridDistortion(),
#                 A.IAAPiecewiseAffine(),
#             ]),
#             A.RandomSizedCrop(min_max_height=(int(resolution*0.7), input_res),
#                               height=resolution, width=resolution, p=1.0),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.GaussianBlur(p=0.3),
#             A.OneOf([
#                 A.RandomBrightnessContrast(),   
#                 A.HueSaturationValue(),
#             ]),
#             A.Cutout(num_holes=8, max_h_size=resolution//8, max_w_size=resolution//8, fill_value=0, p=0.3),
            A.Normalize(),
            ToTensorV2(),
        ], p=1.0)

def get_valid_transforms():
    return A.Compose([
#             A.CenterCrop(height=resolution, width=resolution, p=1.0),
            A.Normalize(),
            ToTensorV2(),
        ], p=1.0)

def get_tta_transforms():
    return A.Compose([
#             A.JpegCompression(p=0.5),
#             A.RandomSizedCrop(min_max_height=(int(resolution*0.9), int(resolution*1.1)),
#                               height=resolution, width=resolution, p=1.0),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.Transpose(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ], p=1.0)


# In[7]:


ds_train = ImageDataset(
    path=TRAIN_ROOT_PATH,
    image_ids=df_train[df_train['fold'] != fold_number].index.values,
    labels=df_train[df_train['fold'] != fold_number].target.values,
    transforms=get_train_transforms(),
)

ds_val = ImageDataset(
    path=TRAIN_ROOT_PATH,
    image_ids=df_train[df_train['fold'] == fold_number].index.values,
    labels=df_train[df_train['fold'] == fold_number].target.values,
    transforms=get_valid_transforms(),
)

ds_test = ImageDataset(
    path=TEST_ROOT_PATH,
    image_ids=df_test.index.values,
    transforms=get_tta_transforms(),
    test=True
)

# del df_train
# len(ds_train), len(ds_val), len(ds_test)


# In[8]:


class MelanomaModel(pl.LightningModule):

    def __init__(self, ds_train, ds_val, output_dim=None):
            
        super(MelanomaModel, self).__init__()
        
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_test = ds_test
        self.val_losses = list()

        self.net = models.resnet18()
        self.net.fc = nn.Linear(in_features=self.net.fc.in_features, out_features=1, bias=True)
        
        
    def forward(self, x):
        x = self.net(x)
        return x
    
    def loss_function(self, y_pred, y_true):
        loss = nn.BCEWithLogitsLoss()
        return loss(y_pred, y_true)
    
    def training_step(self, train_batch, batch_idx):
        x, y_true = train_batch
        y_pred = self.forward(x).flatten()
        loss = self.loss_function(y_pred, y_true.type_as(y_pred))

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        x, y_true = val_batch
        y_pred = self.forward(x).flatten()
        loss = self.loss_function(y_pred, y_true.type_as(y_pred))
        return {'val_batch_loss': loss}

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}] 
        avg_loss = torch.stack([x['val_batch_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        self.val_losses.append(float(avg_loss.cpu().numpy()))
        return {'val_loss': avg_loss, 'log': tensorboard_logs}
    
    def test_step(self, batch, batch_idx):
#         x, _ = batch
#         from IPython.core.debugger import Pdb; Pdb().set_trace()

        logits = self(batch)
        probs = torch.sigmoid(logits)
        
        return {'probs': probs}
    
    def test_epoch_end(self, outputs):
        probs = torch.cat([out['probs'] for out in outputs], dim=0)
        probs = probs.detach().cpu().numpy()
        self.test_predicts = probs  # Save prediction internally for easy access
        # We need to return something 
        return {'dummy_item': 0}
    
    
    def prepare_data(self): 
        pass
    
    def train_dataloader(self): 
        return DataLoader(self.ds_train, batch_size=batch_size, num_workers=8) 

    def val_dataloader(self): 
        return DataLoader(self.ds_val, batch_size=batch_size, num_workers=8) 

    def test_dataloader(self): 
        return DataLoader(self.ds_test, batch_size=batch_size, num_workers=8) 

    def configure_optimizers(self): 
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4) 
        return optimizer 
    


# In[ ]:


early_stop_callback = EarlyStopping(
   monitor='val_loss',
   min_delta=0.00,
   patience=3,
   verbose=False,
   mode='min'
)


# train
model = MelanomaModel(ds_train, ds_val) 

trainer = pl.Trainer(max_epochs=1, early_stop_callback=early_stop_callback, gpus=[1,2])

trainer.fit(model)


# In[ ]:


trainer.test()


# In[ ]:


model.test_predicts


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




