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
from torch.functional import split
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchvision.transforms import transforms
from torchvision import models, datasets, transforms
from PIL import  Image
import matplotlib
import matplotlib.pyplot as plt
import argparse
matplotlib.use( 'tkagg' )

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = argparse.ArgumentParser("to work on dataset, Note, dataset must contain image_id and target columns")

    # -- Add required and optional groups
    parser._action_groups.pop()
    optional = parser.add_argument_group('optional arguments')

    # optional.add_argument("-kd", "--download_from_kaggle", type=bool, default=False,
    #                         help="enable dataset download from kaggle.")
    # optional.add_argument("-kp", "--use_kaggle_processor", type=bool, default=True,
    #                         help="enable dataset download from kaggle.")
    # optional.add_argument("-dn", "--dataset_name", type=str, default="widsdatathon2019.zip",
    #                     help="the initial na of the dataset when downloaded from kaggle")
    return parser


class PalmOilDataSetBackbone(Dataset):
    """Palm Oil Plantation dataset from kaggle, will be used in the PalmOilDataModule."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (str, callable, optional): Optional transform to be applied
                on a sample or any of these ["train", "test", "val"] to pick from
                predefined tranformer.
        """
        #TODO: Load image csv
        self.palmoil_frame = pd.read_csv(csv_file)[:200]

        self.root_dir = root_dir
        
        if transform:
            if transform in ["train", "test", "val"]:
                self._transformer = self._pick_transformer(phase=transform)
            elif type(transform) == transforms.Compose:
                self._transformer = transform
            else:
                "Use default transformer"
                self._transformer = self._pick_transformer()
        else:
            "Pick no transformer"
            self._transformer = None

    def __len__(self):
        return len(self.palmoil_frame)

    def _pick_transformer(self, phase='test'):
        if phase == "each_transform":
            return {
                    "rand_rez_crop" : transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.2, 1)),
                    "rand_ver_flip" : transforms.RandomVerticalFlip(p=0.7),
                    "rand_hor_flip" : transforms.RandomHorizontalFlip(p=0.7),
                    "col_jit" : transforms.ColorJitter(brightness=0.5, contrast=0.05, saturation=0.05, hue=0.05),
                    "rand_rot": transforms.RandomRotation(30),
                    "to_tensor" : transforms.ToTensor(),
                    "norm" : transforms.Normalize((0.5, 0.5, 0.5), 
                                            (0.5, 0.5, 0.5))
            }
        elif phase == 'train':
            return transforms.Compose([ 
                        transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.2, 1)),
                        transforms.RandomVerticalFlip(p=0.7),
                        transforms.RandomHorizontalFlip(p=0.7),
                        transforms.ColorJitter(brightness=0.5, contrast=0.05, saturation=0.05, hue=0.05),
                        transforms.RandomRotation(30),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), 
                                            (0.5, 0.5, 0.5))
                    ])
        
        elif phase == 'val':
                
            return transforms.Compose([ transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), 
                                                                (0.5, 0.5, 0.5))
                                            ])
                        
        elif phase == 'test':

            return transforms.Compose([ transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), 
                                                                (0.5, 0.5, 0.5))
                                            ])
        else:
            raise ValueError("Selected Transformer not available.")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #TODO: extract name from csv
        #TODO:  extract label from csv
        #TODO: extract confident score from csv

        img_name = os.path.join(self.root_dir,self.palmoil_frame.iloc[idx, 0])
        try:
            image = Image.open(img_name)
        except (IOError, OSError) as e:
            try:
                name, ext = self.palmoil_frame.iloc[idx, 0].split(".")
                new_name = name[:-4] + "." + ext
                # print(self.palmoil_frame.iloc[idx, 0], new_name, self.palmoil_frame.iloc[idx, 1], self.palmoil_frame.iloc[idx, 2])
                img_name = os.path.join(self.root_dir,new_name)
                image = Image.open(img_name)
            except (IOError, OSError) as e:
                raise e


        if self._transformer:
            image = self._transformer(image).float()

        y = self.palmoil_frame.iloc[idx, 1]
        score = self.palmoil_frame.iloc[idx, 2]
        target = torch.tensor(data = [y.astype('float32'),score.astype('float32')], dtype=torch.float)


        return (image, target)

    def display(self, idx=0, pick_tranformer="rand_rez_crop"):
        transformers = self._pick_transformer(phase='each_transform')
        assert (idx >= 0) and (idx <= len(self)), f" Index number {idx} not found, available index are 0 - {len(self)}."
        assert pick_tranformer in transformers.keys(), f"Invalid transformer {pick_tranformer}, a list of available transformers are {list(transformers.keys())}."

        image, (target) = self[idx]
        picked_transformer = transformers[pick_tranformer]

        if pick_tranformer not in ["to_tensor", "norm"]:
            transformed_image = picked_transformer(image)
            # Apply transfroms
            fig = plt.figure()

            #plot original
            ax = plt.subplot(1, 2, 0 + 1)
            plt.tight_layout()
            ax.set_title("Original", fontdict={'fontsize': 60, 'fontweight': 60})
            ax.axis('off')
            plt.imshow(image)
            plt.pause(0.001)  # pause a bit so that plots are updated

            #plot transform
            ax = plt.subplot(1, 2, 1 + 1)
            plt.tight_layout()
            ax.set_title(type(picked_transformer).__name__, fontdict={'fontsize': 60, 'fontweight': 60})
            ax.axis('off')
            plt.imshow(transformed_image)
            plt.pause(0.001)  # pause a bit so that plots are updated

            plt.show()
            # print("display")

        elif pick_tranformer == "to_tensor":
            transformed_image = picked_transformer(image)
            print(f"\n\t\t\t\b<<< to_tensor output >>>:\n {transformed_image} \n\n")

        elif pick_tranformer == "norm":
            toTensor_transformer = transformers["to_tensor"]
            image_tensor = toTensor_transformer(image)
            transformed_image = picked_transformer(image_tensor)
            print(f"\n\t\t\t\b<<< normalization output >>>:\n {transformed_image} \n\n")

    def check_missing_data(self, start_idx=0, stop_idx=-1):
        stop_idx = len(self) if stop_idx <= -1 else stop_idx
        self.no_missing = 0
        self.missing = []

        for idx in range(start_idx, stop_idx):
            img_name = os.path.join(self.root_dir,self.palmoil_frame.iloc[idx, 0])
            try:
                image = Image.open(img_name)
            except (IOError, OSError) as e:
                try:
                    name, ext = self.palmoil_frame.iloc[idx, 0].split(".")
                    new_name = name[:-4] + "." + ext
                    img_name = os.path.join(self.root_dir,new_name)
                    image = Image.open(img_name)
                except (IOError, OSError) as e:
                    self.no_missing += 1
                    self.missing.append(idx)
        return self.no_missing, self.missing

    def save_dset(self, name=None):
        self.palmoil_frame.to_csv(name, index=False)

    def remove_all_missing(self, save_as=None):
        self.palmoil_frame = self.palmoil_frame.drop(self.missing)
        self.save_dset(name=save_as)
        self.no_missing, self.missing = 0, []

class PalmOilDataSetModule(pl.LightningDataModule):
    def __init__(self, backbone=PalmOilDataSetBackbone,
                train_batch_size=5, val_batch_size=5, test_batch_size=5, 
                train_num_workers=2, val_num_workers=2, test_num_workers=2, 
                train_root_dir="./dataset/processed/train_images", train_csv="./dataset/processed/traininglabels2.csv", 
                val_root_dir="./dataset/processed/leaderboard_test_data", val_csv="./dataset/processed/testlabels2.csv", 
                test_root_dir="./dataset/processed/leaderboard_holdout_data", test_csv="./dataset/processed/holdout2.csv", 
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

            backbone (torch.utils.data.Dataset): it is a custom dataset loader that defines how the dataset is loaded in.
        """
        super().__init__()

        # load backbone
        self.backbone = backbone

         # set roots
        (self.train_root_dir, self.val_root_dir, self.test_root_dir) = (train_root_dir, val_root_dir, test_root_dir)

        # set csv
        (self.train_csv, self.val_csv, self.test_csv) = (train_csv, val_csv, test_csv)

        # set bacth_size
        (self.train_batch_size, self.val_batch_size, self.test_batch_size) = (train_batch_size, val_batch_size, test_batch_size)

        # set num_workers
        (self.train_num_workers,self.val_num_workers,self.test_num_workers) = (train_num_workers,val_num_workers,test_num_workers)


    def setup(self, stage):
        # transforms for images
        pass

    def train_dataloader(self):
        train_data = self.backbone(csv_file=self.train_csv, root_dir=self.train_root_dir, transform="train")
        return DataLoader(train_data, self.train_batch_size, shuffle=True, num_workers=self.train_num_workers)

    def val_dataloader(self):
        val_data = self.backbone(csv_file=self.val_csv, root_dir=self.val_root_dir, transform="val")
        return DataLoader(val_data, self.val_batch_size,shuffle=False, num_workers=self.val_num_workers)

    def test_dataloader(self):
        test_data = self.backbone(csv_file=self.test_csv, root_dir=self.test_root_dir, transform="test")
        return DataLoader(test_data, self.test_batch_size ,shuffle=False, num_workers=self.test_num_workers)

def main(args):
    # make a PalmOilDataModule instance also passing PalmOilDataSetBackbone as the backbone
    PalmOilDataSet = PalmOilDataSetBackbone(csv_file="./dataset/processed/traininglabels2.csv", root_dir="./dataset/processed/train_images", transform=None)
    # PalmOilDataSet.save_dset(name="outcheck.csv")
    PalmOilDataSet.display(idx=95, pick_tranformer="rand_rez_crop")
    # m,n = PalmOilDataSet.check_missing_data()
    # print(m,n)
    # PalmOilDataSet.remove_all_missing(save_as="./dataset/processed/holdout2.csv")

    # data = PalmOilDataSetModule(train_batch_size=32, train_num_workers=8)


if __name__ == "__main__":

    #Program Flags to control the actions of the program
    args = build_argparser().parse_args()
    main(args)