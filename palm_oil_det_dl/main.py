#load all your packages
# import data manipulation libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

#import frameworks
import torch
# import tensorflow as tf
# import pytorch_lightning as pi

#image manipulation
import zipfile, os
import torch as tch
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.preprocessing import OneHotEncoder

from process import PalmOilDataSetModule
from model import *
import numpy as np
import PIL

import csv
import time
import argparse
from PIL import  Image
from IPython.display import clear_output
from IPython.core.debugger import set_trace
from pytorch_lightning.callbacks import ModelCheckpoint

#parser
def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = argparse.ArgumentParser("Train a model")

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    required.add_argument("-tc", "--train_csv", type=str,
                            default="./dataset/processed/traininglabels2.csv",
                            help="csv file that contains images names to train on.")

    required.add_argument("-td", "--train_rootdir", type=str,
                            default="./dataset/processed/train_images",
                            help="folder path that contains images to train on.")

    required.add_argument("-vd", "--val_rootdir", type=str,
                            default="./dataset/processed/leaderboard_test_data",
                            help="folder path that contains images to val on.")

    required.add_argument("-vc", "--val_csv", type=str,
                            default="./dataset/processed/testlabels2.csv",
                            help="csv file that contains images names to val on.")

    optional.add_argument("-tm", "--trainmode", type=str, default="start",
                            help="mode to train with start:starts a new train/last:picks last \
                            saved/best:picks best/choice:locate check point using name specified.")

    optional.add_argument("-d", "--is_cpu", type=int, default=-1,
                        help="0 for cpu -1 for gpu")

    optional.add_argument('-me', "--max_epoch", type=int, default=3, help="number of epochs to train for.")
    optional.add_argument('-md', "--model_dir", type=str, default='./models/', help="directory where the model checkpoints are stored.")
    optional.add_argument('-fn', "--filename", type=str, default='palm-oil-model4-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}_', help="name to save the model as.")
    optional.add_argument('-cp', "--path_to_ckpt_checkpoint", type=str, default='./models/palm-oil-model-val_loss=0.3733-val_acc=0.9400_val-v4.ckpt', help="load from this check point.")
    optional.add_argument("-pe","--print_stats_every_nepoch",type=int,default=1,help="print train and val statistis every n epoch")
    return parser

def main(args=None):
    # create Instance of the Object Autoencoder

    if args.trainmode == "start":
        print('Starting a new training ...')
        # train
        # model = ResnetBackbone(downlaod_resnet_pretrained=False)
        
        # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
        checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.model_dir,
        filename=args.filename+"val",
        save_top_k=1,
        mode='min',
    )
        classifier = PalmOilLightningClassifier(download_resnet_pretrained=True)
        trainer = Trainer(max_epochs=args.max_epoch, 
                    check_val_every_n_epoch=1,
                    gpus=args.is_cpu,
                    reload_dataloaders_every_epoch=True,
                    callbacks=[checkpoint_callback]
                    )
    elif args.trainmode == "continue":
        print('Continue training from last checkpoint ...')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',dirpath=args.model_dir, filename=args.filename+"val",save_top_k=1,mode='min')
        
        #load from state dict
        classifier = PalmOilLightningClassifier.load_from_checkpoint(args.path_to_ckpt_checkpoint)
        trainer = Trainer(max_epochs=args.max_epoch, 
                check_val_every_n_epoch=1,
                gpus=args.is_cpu,
                reload_dataloaders_every_epoch=True,
                callbacks=[checkpoint_callback]
                )
        
    data = PalmOilDataSetModule(train_num_workers=8, val_num_workers=8,
                                train_batch_size=6, val_batch_size=5, 
                                test_batch_size=5
                                )


    start = time.time()
    trainer.fit(classifier, data)
    print('model finish training at {}'.format(time.time()-start))


if __name__ == "__main__":
# reset && python3 train.py -tdp dataset/labels/test_new1.csv -tdr dataset/Images/test1/ -vdr dataset/Images/test2/ -vdp dataset/labels/test_new2.csv -tm models/other_models/modelsmodel13_train.pt -vm models/other_models/modelsmodel12_val.pt -mon 'modelsmodel14'
    args = build_argparser().parse_args()
    main(args)