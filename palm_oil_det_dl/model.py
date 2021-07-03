#model creation
# import libraries
import numpy as numpy
import pandas as pd
import matplotlib.pyplot as pyplot
# from helper import specificity
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, auc
from IPython.display import clear_output
from torchvision.transforms import transforms
from torchvision import datasets
from torchvision.datasets import MNIST
# from torchsummary import summary
from pytorch_lightning import Trainer
from torch.nn import functional as F
from torch import nn
from torch.nn import Sequential,Dropout, Linear, Identity, Conv2d,MaxPool2d, Conv2d, BatchNorm2d, ReLU
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.optim import SGD,RMSprop
from torchvision import models
from process import PalmOilDataSetBackbone
import pytorch_lightning.metrics.functional as plm
import pytorch_lightning as pl
import torchvision as tv
import torch as tch
import argparse
import torch
import os

 
# This defines the model architecture and forward pass
class ResnetBackbone(nn.Module):

    def __init__(self, downlaod_resnet_pretrained=False):
        super().__init__()

        # mnist images are (1, 28, 28) (channels, width, height) 
        # resnet pretrained model
        self.model = models.resnet50(pretrained=downlaod_resnet_pretrained)

        # edit output
        self.model.fc =  Sequential(OrderedDict([
                                          ('drop', Dropout(p=0.1)),
                                          ('cassifier1', Linear(2048,2)),
        ]))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        return  self.model(x).view(x.shape[0],-1)

#building the pytorch lighting class Model
class PalmOilLightningClassifier(pl.LightningModule):
    def __init__(self, model_backbone=None, dset_backbone=PalmOilDataSetBackbone,
                train_batch_size=32, val_batch_size=32, test_batch_size=32, 
                train_num_workers=2, val_num_workers=2, test_num_workers=2, 
                train_root_dir="./dataset/processed/train_images", train_csv="./dataset/processed/traininglabels.csv", 
                val_root_dir="./dataset/processed/leaderboard_test_data", val_csv="./dataset/processed/testlabels.csv", 
                test_root_dir="./dataset/processed/leaderboard_holdout_data", test_csv="./dataset/processed/holdout.csv", 
                ):
        """
        Args:
            train_batch_size (int): batch size to load image during training.
            val_batch_size (int): batch size to load image during validation.
            test_batch_size (int): batch size to load image during testing.

            train_num_workers (int): num of worker to load dataset with during training.
            val_num_workers (int): num of worker to load dataset with during validation.
            test_num_workers (int): num of worker to load dataset with during testing.

            train_root_dir (string): Path to the croot directory where the training images are contained.
            val_root_dir (string): Path to the croot directory where the validation images are contained.
            test_root_dir (string): Path to the croot directory where the testing images are contained.
                
            train_csv (string): Path to the csv file with training annotations.
            val_csv (string): Path to the csv file with validation annotations.
            test_csv (string): Path to the csv file with testing annotations.

            model_backbone (nn.Module): it is a custom model loader that defines how the model computes feed forward.
            dset_backbone (torch.utils.data.Dataset): it is a custom dataset loader that defines how the dataset is loaded in.
        """
        super().__init__()
        self.model_backbone = model_backbone
        
         # set roots
        (self.train_root_dir, self.val_root_dir, self.test_root_dir) = (train_root_dir, val_root_dir, test_root_dir)

        # set csv
        (self.train_csv, self.val_csv, self.test_csv) = (train_csv, val_csv, test_csv)

        # set bacth_size
        (self.train_batch_size, self.val_batch_size, self.test_batch_size) = (train_batch_size, val_batch_size, test_batch_size)

        # set num_workers
        (self.train_num_workers,self.val_num_workers,self.test_num_workers) = (train_num_workers,val_num_workers,test_num_workers)

        # load backbone
        self.dset_backbone = dset_backbone

    def configure_optimizers(self):
        return RMSprop([{'params' : self.parameters, 'lr':0.0000002}],
                       lr=0.0000002, momentum=0.9)

    def my_loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def training_step(self, train_batch, batch_idx):
        # unpack
        x,y = train_batch

        # forward pass
        logits = self.backbone(x)
        
        #get predictions and loss
        _, preds = torch.max(logits, 1)
        loss = self.my_loss(logits, y)
        
        self.log("train_loss", loss, on_epoch=True)
        return {'loss' : loss,'y_pred':preds.cpu(), 'y':y.cpu()}

    def validation_step(self, val_batch, batch_idx):
        # unpack
        x,y = val_batch

        # forward pass  
        logits = self.backbone(x)
        
        # get predictions and loss
        _, preds = torch.max(logits, 1)
        loss = self.my_loss(logits, y)
        
        self.log("val_loss", loss)
        return {'val_loss' :loss,'y_pred':preds.cpu(), 'y':y.cpu()}

    def test_step(self, test_batch, batch_idx):
        # unpack
        x,y = test_batch

        # forward pass  
        logits = self.backbone(x)
        
        # get predictions and loss
        _, preds = torch.max(logits, 1)
        loss = self.my_loss(logits, y)
        
        self.log("test_loss", loss)
        return {'test_loss' :loss,'y_pred':preds.cpu(), 'y':y.cpu()}

    def validation_epoch_end(self, outputs):
        pass

    def test_epoch_end(self, outputs):
        pass

    def prepare_data(self):
         pass

    def download_dset(self):
        pass

    def train_dataloader(self, transform=None):
        transform = transform if transform else self.trans["train"]
        assert type(transform) == type(transforms.Compose), "Train expected transform of type transforms.Compose"

        train_data = self.dset_backbone(csv_file=self.train_csv, root_dir=self.train_root_dir, transform="train")
        return DataLoader(train_data, self.train_batch_size, shuffle=True, num_workers=self.train_num_workers)

    def val_dataloader(self, transform=None):
        transform = transform if transform else self.trans["val"]
        assert type(transform) == type(transforms.Compose), "Val expected transform of type transforms.Compose"

        val_data = self.dset_backbone(csv_file=self.val_csv, root_dir=self.val_root_dir, transform="val")
        return DataLoader(val_data, self.val_batch_size,shuffle=False, num_workers=self.val_num_workers)

    def test_dataloader(self, transform=None):
        transform = transform if transform else self.trans["test"]
        assert type(transform) == type(transforms.Compose), "Test expected transform of type transforms.Compose"

        test_data = self.dset_backbone(csv_file=self.test_csv, root_dir=self.test_root_dir, transform="test")
        return DataLoader(test_data, self.test_batch_size ,shuffle=False, num_workers=self.test_num_workers)