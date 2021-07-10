#model creation
# import libraries
import numpy as numpy
import pandas as pd
import matplotlib.pyplot as pyplot
# from helper import specificity
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, auc
from IPython.display import clear_output
# from pytorch_lightning.metrics.functional.classification import f1_score, precision
from torchvision.transforms import transforms
from torchvision import datasets
from torchvision.datasets import MNIST
# from torchsummary import summary
from pytorch_lightning import Trainer
from torch.nn import functional as F
from torch import nn
from torch.nn import Sequential,Dropout, Linear, Identity, Conv2d,MaxPool2d, Conv2d, BatchNorm2d, ReLU, Softmax, Sigmoid
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.optim import SGD,RMSprop
from torchvision import models
# import pytorch_lightning.metrics.functional as plm
import pytorch_lightning.metrics.functional as plm
from torchmetrics import Accuracy, F1
# import pytorch_lightning.metrics.functional as FM
# from pytorch_lightning.metrics.functional import f1_score, accuracy, recall, precision
import pytorch_lightning as pl
import torchvision as tv
import torch as tch
import argparse
import torch
import os

 
# This defines the model architecture and forward pass
class ResnetBackbone(nn.Module):

    def __init__(self, download_resnet_pretrained=False):
        super().__init__()

        # mnist images are (1, 28, 28) (channels, width, height) 
        # resnet pretrained model
        self.model = models.resnet50(pretrained=download_resnet_pretrained)

        # edit output
        self.model.fc =  Sequential(OrderedDict([
                                          ('drop', Dropout(p=0.3)),
                                          ('cassifier1', Linear(2048,1)),
        ]))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        return  self.model(x).view(x.shape[0],-1)

#building the pytorch lighting class Model
class PalmOilLightningClassifier(pl.LightningModule):
    def __init__(self, download_resnet_pretrained=False):
        """
        Args:

            backbone (nn.Module): it is a custom model loader that defines how the model computes feed forward.
        """
        super(PalmOilLightningClassifier, self).__init__()
        self.acc = Accuracy()
        # self.f1_score = F1(num_classes=2)
        # self.save_hyperparameters()

        # mnist images are (1, 28, 28) (channels, width, height) 
        # resnet pretrained model
        self.model = models.resnet50(pretrained=download_resnet_pretrained)

        # edit output
        self.model.fc =  Sequential(OrderedDict([
                                          ('drop', Dropout(p=0.3)),
                                          ('relu', ReLU()),
                                          ('cassifier1', Linear(2048,1)),
                                          ('sig', Sigmoid())

        ]))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        return  self.model(x).view(x.shape[0],-1)

    def configure_optimizers(self):
        return SGD(self.model.parameters(), lr=0.0003)

    def my_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx,pred_threshold=0.5):
        loss, acc = self.share_step(batch=batch,pred_threshold=pred_threshold)
        metrics = {
                    'loss': loss,
                    'train_acc': acc,
                }
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return metrics

    def validation_step(self, batch, batch_idx,pred_threshold=0.5):
        loss, acc = self.share_step(batch=batch,pred_threshold=pred_threshold)
        metrics = { 'val_loss': loss, 'val_acc': acc}
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return metrics

    def validation_step_end(self, val_step_outputs):
        pass

    def test_step(self, batch, batch_idx,pred_threshold=0.10):
        metrics = self.validation_step(batch, batch_idx,pred_threshold)
        metrics = {'test_acc': metrics['val_acc'], 'test_loss': metrics['val_loss']}
        self.log_dict(metrics)

    def share_step(self, batch, pred_threshold=0.5):
        # unpack
        x,y = batch

        # forward pass
        pred_probab = self.model(x)
        
        #get predictions and loss
        # pred_probab = nn.Softmax(dim=1)(logits)
        # pred_probab = nn.Sigmoid()(logits)
        # print(pred_probab)
        # y_hat = pred_probab.argmax(1)
        y_hat = (pred_probab.clone().detach() > pred_threshold).type(torch.int)
        # print(pred_probab, y[:,:1])
        loss = self.my_loss(pred_probab, y[:,:1])
        # print(y_hat, y[:,:1].long())
        acc = self.acc(y_hat, y[:,:1].long())
        return loss, acc

    # def on_save_checkpoint(checkpoint):
    #     # 99% of use cases you don't need to implement this method
    #     # checkpoint['something_cool_i_want_to_save'] = my_cool_pickable_object
    #     pass