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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from process_data import torch_dset
from model import *
import numpy as np
import PIL

import csv
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

    required.add_argument("-i", "--input", required=True, type=str,
                            help="csv file that contains images names to train on.")

    required.add_argument("-r","--root", required=True, type=str,
                            help="path to folder where the images are contained.")

    optional.add_argument("-vm", "--valmodel", type=str, default=None,
                            help="path to .pt or pth file with a trained model.")

    optional.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")

    optional.add_argument("-d", "--is_cpu", type=int, default=0,
                        help="0 for cpu -1 for gpu")

    optional.add_argument("-o","--output",type=str,default='.',help="saves output to local machine where the whole network runs")
    optional.add_argument("-pe","--print_stats_every_nepoch",type=int,default=1,help="print train and val statistis every n epoch")
    return parser

def main(args=None):
    # load model
    print('Continue training from checkpoint ...')
    states = torch.load(args.valmodel,map_location='cpu')
    model = Dia_ret(every_b=args.print_stats_every_nepoch,best_acc=states['best_acc'],curr_ephochs=states['epochs'],train_flag=False, test_dataroot=args.root, test_datapath=args.input,)
    model.load_state_dict(states['state_dict_cnn'])

    #load dataset
    if args.input.split('.')[-1] in ['jpg','png','jpeg']:
        val_trans = transforms.Compose([ transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), 
                                                             (0.5, 0.5, 0.5))
                                          ])
        img = val_trans(Image.open(args.input))
        model.eval()
        start = time.time()
        out = model(img)
        print('model finish testing at {}'.format(1000*(time.time()-start)))

    elif args.input.split('.')[-1] in ['csv']:
        trainer = Trainer(max_epochs=1, 
                    check_val_every_n_epoch=1,
                    gpus=args.is_cpu,
                    reload_dataloaders_every_epoch=True,
                    )

        start = time.time()
        trainer.test(model)
        print('model finish testing at {}ms'.format(1000*(time.time()-start)))

if __name__ == "__main__":
# reset && python3 test.py -i dataset/labels/test_new2.csv -r dataset/Images/test2/ -vm models/other_models/modelsmodel13_train.pt
    args = build_argparser().parse_args()
    main(args)