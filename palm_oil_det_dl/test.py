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
import zipfile, os,time
import torch as tch
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from process import PalmOilDataSetModule
from model import *
import numpy as np
import PIL

import csv
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

    required.add_argument("-td", "--test_rootdir", type=str,
                            default="./dataset/processed/leaderboard_holdout_data",
                            help="folder path that contains images to test on.")

    required.add_argument("-vc", "--val_csv", type=str,
                            default="./dataset/processed/holdout2.csv",
                            help="csv file that contains images names to test on.")

    optional.add_argument("-tm", "--testmode", type=str, default="kaggle",
                            help="mode to train with start:starts a new train/last:picks last \
                            saved/best:picks best/choice:locate check point using name specified.")

    optional.add_argument("-d", "--is_cpu", type=int, default=0,
                        help="0 for cpu -1 for gpu")

    optional.add_argument('-cp', "--path_to_ckpt_checkpoint", type=str, default='./models/palm-oil-model3-epoch=00-val_loss=0.2185-val_acc=0.9358_val.ckpt', help="load from this check point.")
    optional.add_argument("-pe","--print_stats_every_nepoch",type=int,default=1,help="print train and val statistis every n epoch")
    return parser


def main(args=None):
    # load model
    print('Continue training from checkpoint ...')
    mapping = {'cuda:0':'cpu'} if args.is_cpu == 0 else {'cuda:0':'cuda:0'}
    classifier = PalmOilLightningClassifier.load_from_checkpoint(args.path_to_ckpt_checkpoint, map_location=mapping)

    #load dataset
    if args.testmode == "kaggle":
        
        #load from state dict
        trainer = Trainer(gpus=args.is_cpu)
        data = PalmOilDataSetModule(train_num_workers=8, val_num_workers=8,
                                train_batch_size=6, val_batch_size=5, 
                                test_batch_size=5
                                )
        start = time.time()
        trainer.test(classifier, data.val_dataloader())
        print('model finish testing at {}ms'.format(1000*(time.time()-start)))

    elif args.input.split('.')[-1] in ['jpg','png','jpeg'] and args.testmode == 'single':
        print("Single mode testing....")
        val_trans = transforms.Compose([ transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), 
                                                             (0.5, 0.5, 0.5))
                                          ])
        img = val_trans(Image.open(args.input))
        classifier.eval()
        start = time.time()
        out = classifier(img)
        print('model finish testing at {}'.format(1000*(time.time()-start)))

    elif args.input.split('.')[-1] in ['csv'] and args.testmode == "multi":
        pass

if __name__ == "__main__":
# reset && python3 test.py -i dataset/labels/test_new2.csv -r dataset/Images/test2/ -vm models/other_models/modelsmodel13_train.pt
    args = build_argparser().parse_args()
    main(args)