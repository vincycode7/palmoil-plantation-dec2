#load all your packages
# import data manipulation libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

#import frameworks
import torch as tch
# import tensorflow as tf
# import pytorch_lightning as pi

#image manipulation
import zipfile, os
import torch as tch
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.preprocessing import OneHotEncoder

from process import PalmOilDataSetBackbone
from model import *
import numpy as np
import PIL

import csv
import time
import argparse
from PIL import  Image
from IPython.display import clear_output
from IPython.core.debugger import set_trace

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
                            default="./dataset/processed/traininglabels.csv",
                            help="csv file that contains images names to train on.")

    required.add_argument("-td", "--train_rootdir", type=str,
                            default="./dataset/processed/train_images",
                            help="folder path that contains images to train on.")

    required.add_argument("-vd", "--val_rootdir", type=str,
                            default="./dataset/processed/leaderboard_test_data",
                            help="folder path that contains images to val on.")

    required.add_argument("-vc", "--val_csv", type=str,
                            default="./dataset/processed/testlabels.csv",
                            help="csv file that contains images names to val on.")

    optional.add_argument("-tm", "--trainmodel", type=str, default=None,
                            help="path to .pt or pth file with a trained model.")
    optional.add_argument("-vm", "--valmodel", type=str, default=None,
                            help="path to .pt or pth file with a trained model.")

    optional.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")

    optional.add_argument("-d", "--is_cpu", type=int, default=0,
                        help="0 for cpu -1 for gpu")

    optional.add_argument('-mp', "--max_epoch", type=int, default=1, help="number of epochs to train for.")
    optional.add_argument('-mon', "--model_out_name", type=str, default='model', help="name to save the model as.")
    optional.add_argument("-o","--output",type=str,default='.',help="saves output to local machine where the whole network runs")
    optional.add_argument("-pe","--print_stats_every_nepoch",type=int,default=1,help="print train and val statistis every n epoch")
    return parser

def main(args=None):
    # create Instance of the Object Autoencoder
    bst_acc = 0 if not args.valmodel else torch.load(args.valmodel,map_location='cpu')['best_acc']

    if not args.trainmodel:
        print('Starting a new training ...')
        # train
        model = ResnetBackbone()
    else:
        print('Continue training from checkpoint ...')

        # load state
        states = torch.load(args.trainmodel,map_location='cpu')
        model = ResnetBackbone()

        #load from state dict
        model.load_state_dict(states['state_dict_cnn'])
        
    classifier = PalmOilLightningClassifier(model, train_root_dir="./dataset/processed/train_images", train_csv="./dataset/processed/traininglabels.csv", 
                val_root_dir="./dataset/processed/leaderboard_test_data", val_csv="./dataset/processed/testlabels.csv")

    trainer = Trainer(max_epochs=args.max_epoch, 
                    check_val_every_n_epoch=1,
                    gpus=args.is_cpu,
                    reload_dataloaders_every_epoch=True
                    )

    start = time.time()
    trainer.fit(classifier, data)
    print('model finish training at {}'.format(time.time()-start))


if __name__ == "__main__":
# reset && python3 train.py -tdp dataset/labels/test_new1.csv -tdr dataset/Images/test1/ -vdr dataset/Images/test2/ -vdp dataset/labels/test_new2.csv -tm models/other_models/modelsmodel13_train.pt -vm models/other_models/modelsmodel12_val.pt -mon 'modelsmodel14'
    args = build_argparser().parse_args()
    main(args)