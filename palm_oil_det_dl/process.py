#import data manipulation libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
# import py7zr
# import parser


#data loader
# !pip install pytorch-lightning
# !pip install torchsummary
# !pip install pytorch-model-summary
import os, torch
import torch as tch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchvision.transforms import transforms
from PIL import  Image
import matplotlib.pyplot as plt
import argparse


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = argparse.ArgumentParser("to work on dataset, Note, dataset must contain image_id and target columns")

    # -- Add required and optional groups
    parser._action_groups.pop()
    optional = parser.add_argument_group('optional arguments')

    optional.add_argument("-kd", "--download_from_kaggle", type=bool, default=False,
                            help="enable dataset download from kaggle.")
    optional.add_argument("-kp", "--use_kaggle_processor", type=bool, default=True,
                            help="enable dataset download from kaggle.")
    optional.add_argument("-dn", "--dataset_name", type=str, default="widsdatathon2019.zip",
                        help="the initial na of the dataset when downloaded from kaggle")
    
    # optional.add_argument('-mp',"--maps_", type=dict, default={0:0, 1:1, 2:2, 3:2, 4:2})
    # optional.add_argument("-cl", "--change_label", type=bool, default=False,
    #                         help="to change the label of dataset.")
    # optional.add_argument("-lp", "--label_path", type=str, default=None,
    #                     help="path where dataset is present")

    # optional.add_argument("-lo", "--new_label_out_name", type=str, default=None,
    #                     help="name to give new label is shuffle slipt is not enabled")

    # optional.add_argument('-ss', "--to_shuffle_split", type=bool, default=False, help="to shuffle and split data into train test val")

    # optional.add_argument('-sn', "--shuff_new_label_out_name", type=tuple, default=None, help="name to save the shuffle and splited dataset.")
    # optional.add_argument("-sd","--save_ds",type=bool,default=False,help="if to save dataset")
    return parser

class PALMOILDataModule(Dataset):

  #initialize the dataloader instance
  def __init__(self, csv_file=None, train=True, X=None, y=None,root='.'):
    if csv_file:
        try:
            data = pd.read_csv(root+csv_file)
            self.X, self.y = data.image_id,data.target
        except Exception as e:
            print(e)
    else:
        try:
            self.X,self.y = pd.read_csv(root+X), pd.read_csv(root+y)
        except Exception as e:
            print(e)
    
    self.root = root
    trans = {'train': transforms.Compose([ 
                      transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.2, 1)),
                      transforms.RandomVerticalFlip(),
                      transforms.RandomHorizontalFlip(),
                      transforms.ColorJitter(brightness=0.5, contrast=0.05, saturation=0.05, hue=0.05),
                      transforms.RandomRotation(30),
                      transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), 
                                          (0.5, 0.5, 0.5))
                  ]),
             
                'val' : transforms.Compose([ transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), 
                                                             (0.5, 0.5, 0.5))
                                          ])
              }
    self.trans = trans['train'] if train else trans['val']

  def __getitem__(self, idx):
    #check if last sample 
    #     if idx == (self.__len__()):raise StopIteration
    if torch.is_tensor(idx):idx = idx.tolist() 

    try:
        x_,y_ = self.root+self.X[idx], self.y[idx]
        img = self.trans(Image.open(x_)).float()
    except:
        try:
            x_,y_ = self.root+self.X[idx]+'.jpg', self.y[idx]
            img = self.trans(Image.open(x_)).float()
        except:
            try:
                x_,y_ = self.root+self.X[idx]+'.png', self.y[idx]
                img = self.trans(Image.open(x_)).float()
            except:
                try:
                    x_,y_ = self.root+self.X[idx]+'.jpeg', self.y[idx]
                    img = self.trans(Image.open(x_)).float()
                except Exception as e:
                    print(e)

    #open images and resize
    y_ = torch.tensor(y_.astype('float32')).long()
    return img, y_

  def __len__(self):
    return len(self.X)

def main(args):
    # download dataset

    #unpack the dataset

    #make copy of dataset

    #move copy to a defferent folder

    # rename copy

    pass


if __name__ == "__main__":

    #Program Flags to control the actions of the program
    args = build_argparser().parse_args()
    main(args)