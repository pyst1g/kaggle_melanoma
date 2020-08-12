#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pytorch_lightning as pl
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import log_loss, roc_auc_score
from torchvision import models, transforms

# from sync_batchnorm import convert_model


# In[3]:


class ConcatModel(pl.LightningModule):

    def __init__(self,image_model,  ds_train, ds_val,ds_test, n_features, batch_size=64,):
            
        super(ConcatModel, self).__init__()
        
        self.image_model = image_model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_test = ds_test
        self.batch_size = batch_size
        self.input_dim = n_features
        self.val_logloss = dict()
        self.val_auc = dict()
        self.oof_preds = dict()
            
        # テーブルデータ用
        self.layer1_1 = torch.nn.Linear(n_features, 64)    
        self.bn1_1 = nn.BatchNorm1d(64)
        self.layer1_2 = torch.nn.Linear(64, 32)
        
        #　 結合データ用
        self.layer2_1 = torch.nn.Linear(self.image_model.n_map_features + self.layer1_2.out_features, 1024)
        self.bn2_1 = nn.BatchNorm1d(1024)
        self.layer2_2 = torch.nn.Linear(1024, 512)
        self.bn2_2 = nn.BatchNorm1d(512)
        self.layer2_3 = torch.nn.Linear(512, 256)
        self.bn2_3 = nn.BatchNorm1d(256)
        self.fc = torch.nn.Linear(256, 1)
        
        self.val_losses = list()

        
    def forward(self, image, x):
        
        # 画像の特徴量抽出
        self.image_model.eval()
        with torch.no_grad():
            x1 = self.image_model(image, feature_extract=True) 
        
        # tableの特徴量抽出
        x2 = self.table_forward(x)
        
        #　くっつける
        x_cat = torch.cat([x1,x2], dim=1)
        
        outputs = self.concat_forward(x_cat)
        
        
        return outputs
    
    def table_forward(self, x):
#         x = x.view(-1, self.input_dim)
        # layer 1-1
        x = self.layer1_1(x)
        x = self.bn1_1(x)
        x = torch.relu(x)
        # layer 1-2
        x = self.layer1_2(x)
        
        return x
        
    def concat_forward(self, x):
#         x = x.view(-1, self.input_dim)
        # layer 2-1
        x = self.layer2_1(x)
        x = self.bn2_1(x)
        x = torch.relu(x)
        # layer 2-2
        x = self.layer2_2(x)
        x = self.bn2_2(x)
        x = torch.relu(x)
        # layer 2-3
        x = self.layer2_3(x)
        x = self.bn2_3(x)
        x = torch.relu(x)
        x = self.fc(x)
        
        return x
    
    def loss_function(self, y_pred, y_true):
        loss = nn.BCEWithLogitsLoss()
        return loss(y_pred, y_true)
    
    def training_step(self, train_batch, batch_idx):
        image, x, y_true = train_batch
        logits = self(image, x)
        y_pred = logits.flatten()
        probs = torch.sigmoid(logits)
        loss = self.loss_function(y_pred, y_true.type_as(y_pred))
        
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs, 'probs': probs, 'y_true':y_true}
    
    def train_epoch_end(self, outputs):
        probs = torch.cat([out['probs'] for out in outputs], dim=0)
        probs = probs.detach().cpu().numpy()
        
        y_true = torch.cat([out['y_true'] for out in outputs], dim=0)
        y_true = y_true.detach().cpu().numpy()
        
        auc_score = roc_auc_score(y_true, probs)
        
        tensorboard_logs = {'train_epoch_loss': cross_entropy, 'train_auc':auc_score}
        return {'log': tensorboard_logs}
    
    def validation_step(self, val_batch, batch_idx):
        image, x, y_true = val_batch
        logits = self(image, x)
        y_pred = logits.flatten()
        probs = torch.sigmoid(logits)
#         loss = self.loss_function(y_pred, y_true.type_as(y_pred))
#         return {'val_batch_loss': loss, 'probs':probs, 'y_true':y_true}
        return {'probs':probs, 'y_true':y_true}

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}] 
        probs = torch.cat([out['probs'] for out in outputs], dim=0)
        probs = probs.detach().cpu().numpy()
        
        y_true = torch.cat([out['y_true'] for out in outputs], dim=0)
        y_true = y_true.detach().cpu().numpy()
        
        auc_score = roc_auc_score(y_true, probs)
        cross_entropy = log_loss(y_true, probs)
        
        tensorboard_logs = {'valid_logloss': cross_entropy, 'valid_auc':auc_score}
        self.val_logloss[self.current_epoch] = cross_entropy
        self.val_auc[self.current_epoch] = auc_score
        self.oof_preds[self.current_epoch] = probs
        return {'val_loss': cross_entropy, 'val_auc': auc_score, 'log': tensorboard_logs}
    
    def test_step(self, batch, batch_idx):
        image, x, _ = batch

        logits = self(image, x)
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
        return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=8, shuffle=True, drop_last=True, pin_memory=True) 

    def val_dataloader(self): 
        return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=8, pin_memory=True) 

    def test_dataloader(self): 
        return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=8, pin_memory=True) 

    def configure_optimizers(self): 
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4) 
        scheduler =  {
         'scheduler': ReduceLROnPlateau(optimizer, factor=0.2, patience=3, mode='max'),
         'monitor': 'val_auc', # Default: val_loss
         'interval': 'epoch',
         'frequency': 1
        }
        return [optimizer], [scheduler]



# In[4]:


class MelanomaModel(pl.LightningModule):

    def __init__(self, model_name, ds_train, ds_val, ds_test, batch_size=64):
            
        super(MelanomaModel, self).__init__()
        
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_test = ds_test
        self.batch_size = batch_size
        self.val_logloss = dict()
        self.val_auc = dict()
        self.oof_preds = dict()

        if "efficient" in model_name:
            self.net = ptcv_get_model(model_name, pretrained=True)
            self.n_map_features = self.net.output.fc.in_features
            self.fc = nn.Linear(in_features=self.net.output.fc.in_features, out_features=1)
            self.net.output.fc = nn.Identity()
        elif "resne" in model_name:
            self.net = models.resnet18(pretrained=True)
            self.n_map_features = list(self.net.children())[-1].in_features
            self.fc = nn.Linear(in_features=self.net.fc.in_features, out_features=1)
            self.net.fc = nn.Identity()
        else:
            raise Exception("未実装")
            
        

        
    def forward(self, x, feature_extract=False):
        x = self.net(x)
        if feature_extract:
            return x
        else:
            x = self.fc(x)
            return x
    
    def loss_function(self, y_pred, y_true):
        loss = nn.BCEWithLogitsLoss()
        return loss(y_pred, y_true)
    
    def training_step(self, train_batch, batch_idx):
        x, y_true = train_batch
        y_pred = self(x).flatten()
        loss = self.loss_function(y_pred, y_true.type_as(y_pred))

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        x, y_true = val_batch
        logits = self(x)
        y_pred = logits.flatten()
        probs = torch.sigmoid(logits)
#         loss = self.loss_function(y_pred, y_true.type_as(y_pred))
#         return {'val_batch_loss': loss, 'probs':probs, 'y_true':y_true}
        return {'probs':probs, 'y_true':y_true}

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}] 
        probs = torch.cat([out['probs'] for out in outputs], dim=0)
        probs = probs.detach().cpu().numpy()
        
        y_true = torch.cat([out['y_true'] for out in outputs], dim=0)
        y_true = y_true.detach().cpu().numpy()
        
        auc_score = roc_auc_score(y_true, probs)
        cross_entropy = log_loss(y_true, probs)
        
        tensorboard_logs = {'valid_logloss': cross_entropy, 'valid_auc':auc_score}
        self.val_logloss[self.current_epoch] = cross_entropy
        self.val_auc[self.current_epoch] = auc_score
        self.oof_preds[self.current_epoch] = probs
        return {'val_loss': cross_entropy, 'val_auc': auc_score, 'log': tensorboard_logs}
    
    def test_step(self, batch, batch_idx):
        x, _ = batch

        logits = self(x)
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
        return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=8, shuffle=True, drop_last=True, pin_memory=True) 

    def val_dataloader(self): 
        return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=8, pin_memory=True) 

    def test_dataloader(self): 
        return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=8, pin_memory=True) 

    def configure_optimizers(self): 
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4) 
        scheduler =  {
         'scheduler': ReduceLROnPlateau(optimizer, factor=0.1, patience=3, mode='max'),
         'monitor': 'val_auc', # Default: val_loss
         'interval': 'epoch',
         'frequency': 1
        }
#         scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min')
        return [optimizer], [scheduler]


# In[ ]:




