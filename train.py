import torch, torch.nn as nn, torch.utils.data as data, torchvision as tv, torch.nn.functional as F
import lightning as L
import os
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.tuner import Tuner
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import albumentations as A 
from os.path import join
from torch.utils.data import DataLoader
import torch.nn.functional as F
import PIL
from PIL import Image
import numpy as np
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from torchmetrics.segmentation import MeanIoU
import torch.optim.lr_scheduler as lr_scheduler 
from lightning.pytorch.callbacks import ModelCheckpoint
import cv2
import pandas as pd
from osgeo import gdal
import json
import seaborn as sns
import torchvision 
import sys
from PIL import Image
import matplotlib
# matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt  
import functools
import builtins
builtins.print = functools.partial(print, flush=True) 
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import JaccardLoss
import models
from models import SMP_SemanticSegmentation, DINOv2_SemanticSegmentation
import tifffile as tiff
from torchvision.transforms import Resize

# tensorboard --logdir=./lightning_logs/
# ctrl shft p -> Python: Launch Tensorboard  select lightning logs
    
class mmsegyrebDataset(Dataset):
    def __init__(self, image_set,  root_dir,  transform=None):
        
        # image_set # train ,test, validation
        self.transform = transform
        self.root = join(root_dir,image_set)
  
        self.sar_dir =  join(self.root, "SAR")
        self.msi_dir =  join(self.root, "MSI")
        self.label_dir = join(self.root, "label")
        
        self.img_names_sar = [f for f in os.listdir(self.sar_dir) if f.endswith('.' + 'tif')]
        self.img_names_msi = [f for f in os.listdir(self.msi_dir) if f.endswith('.' + 'tif')]
        
        try:
            self.img_labels = [f for f in os.listdir(self.label_dir) if f.endswith('.' + 'tif')]
        except:
            self.img_labels = []
        
        
        self.num_images = len( self.img_names_sar) 

        if len(self.img_labels)>0:
            assert self.num_images == len( self.img_labels)
            # sort img_labels
            self.img_labels.sort()
        assert self.num_images == len( self.img_names_msi)
        
        self.img_names_sar.sort()
        self.img_names_msi.sort()

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        
        
        msi_name, ext_msi = os.path.splitext(self.img_names_msi[idx])
        sar_name, ext_sar = os.path.splitext(self.img_names_sar[idx])
        
        assert sar_name == msi_name # make sure they have the same name 
        
        
        if len(self.img_labels)>0:
            label_name, ext_label = os.path.splitext(self.img_labels[idx])
            assert label_name == sar_name # make sure they have the same name
            # read the label image 
            label_path = join(self.label_dir, self.img_labels[idx])
            label_img = Image.open(label_path)
            label_img_np = np.array(label_img) # int32 x,y
            
            # print(label_img_np.shape, label_img_np.dtype, np.min(label_img_np), np.max(label_img_np))
        else :
            label_img_np = np.zeros((256,256), dtype=np.int32)
        # read the msi image
        msi_path = join(self.msi_dir, self.img_names_msi[idx])
        msi_img = tiff.imread(msi_path)
        msi_img_np = np.array(msi_img) # int16 x,y,channels=12
        
        # print(msi_img_np.shape, msi_img_np.dtype, np.min(msi_img_np), np.max(msi_img_np))
        
        # read the sar image
        sar_path = join(self.sar_dir, self.img_names_sar[idx])
        sar_img = tiff.imread(sar_path)
        sar_img_np = np.array(sar_img) # int16 x,y,channels=2
        
        # Add a third channel of zeros to the SAR image
        zeros_channel = np.zeros((sar_img_np.shape[0], sar_img_np.shape[1]), dtype=sar_img_np.dtype)
        sar_img_np = np.dstack((sar_img_np, zeros_channel)) # Now sar_img_np has shape (x, y, 3)
        
        # stack the msi and sar images
        msi_img_np = np.dstack((msi_img_np, sar_img_np)) # Now sar_img_np has shape (x, y, 14)


        # print(sar_img_np.shape, sar_img_np.dtype, np.min(sar_img_np), np.max(sar_img_np))

        # format of images are x,y,channels for albumnetations
        # Convert images from int16 to float32
        msi_img_np = msi_img_np.astype(np.float32)
        sar_img_np = sar_img_np.astype(np.float32)
        
        # print(msi_img_np.shape, sar_img_np.shape, label_img_np.shape, msi_img_np.dtype, sar_img_np.dtype, label_img_np.dtype)
        
        # Apply transformations if any
        if self.transform:
            if len(self.img_labels)>0:
                transformed = self.transform(image=sar_img_np, mask=label_img_np, msi_image=msi_img_np)
            else:
                transformed = self.transform(image=sar_img_np, msi_image=msi_img_np)
                label_img = torch.tensor(label_img_np, dtype=torch.int32)
            msi_img = torch.tensor(transformed['msi_image'], dtype=torch.float32)
            sar_img = torch.tensor(transformed['image'], dtype=torch.float32)
            if len(self.img_labels)>0:
                label_img = torch.tensor(transformed['mask'], dtype=torch.int32)
        else:
            msi_img = torch.tensor(msi_img_np, dtype=torch.float32)
            sar_img = torch.tensor(sar_img_np, dtype=torch.float32)
            label_img = torch.tensor(label_img_np, dtype=torch.int32)
            
            
        # #convert from x,y,channels to channels, x, y
        msi_img = msi_img.permute(2,0,1)
        sar_img = sar_img.permute(2,0,1)
        
        # print(msi_img.shape, sar_img.shape, label_img.shape, msi_img.dtype, sar_img.dtype, label_img.dtype)

            
        return msi_img, sar_img, label_img, self.img_names_msi[idx]   
    

dataset_dir='/workspaces/MMSeg-YREB'
# 9 LULC classes are: 0) Background, 1) Tree, 2) Grassland, 3) Cropland, 4) Low Vegetation, 5) Wetland, 6) Water, 7) Built-up, 8) Bare ground, 9) Snow.
num_classes = 10 # ignore 0 background 
batch_size = 16
ignore_index=0 # background
num_workers = 16 #  os.cpu_count() or 1  # Fallback to 1 if os.cpu_count() is None
initial_lr =  0.001 
swa_lr = 0.01
# these should be multiple of 14 for dino model 
# input image is of size 256x256
img_height = 256
img_width = 256
max_num_epochs = 100
accumulate_grad_batches = 1 # increases the effective batch size  # 1 means no accumulation # more important when batch size is small or not doing multi gpu training
grad_clip_val = 5 # clip gradients that have norm bigger than this
training_model = True
tuning_model = False
test_model = False
# min_epochs = 20

# Define mean and standard deviation for normalization
# Use the same value for all channels
num_channels = 14
mean = [0.45] * num_channels  
std = [0.225] * num_channels   

torch.cuda.empty_cache()

test_transform = A.Compose([
    A.Resize(width=img_width, height=img_height), 
    A.Normalize(mean=mean, std=std, max_pixel_value=255.0)

], additional_targets={"msi_image": "image"})

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Resize(width=img_width, height=img_height), 
    A.Normalize(mean=mean, std=std, max_pixel_value=255.0)
], additional_targets={"msi_image": "image"})


train_dataset = mmsegyrebDataset(image_set="train", root_dir=dataset_dir,  transform=train_transform)


model = SMP_SemanticSegmentation(num_classes=num_classes,learning_rate=initial_lr, ignore_index=ignore_index, num_channels= num_channels, num_workers=num_workers,  train_dataset=train_dataset, batch_size=batch_size)
# model = DINOv2_SemanticSegmentation(num_classes=num_classes,learning_rate=initial_lr, ignore_index=ignore_index, num_channels= num_channels, num_workers=num_workers,  train_dataset=train_dataset,  batch_size=batch_size)


checkpoint_callback_train_loss = ModelCheckpoint(monitor="train_loss", mode="min", save_top_k=1, filename="lowest_train_loss_hsi")
checkpoint_callback_train_miou = ModelCheckpoint(monitor="train_miou", mode="max", save_top_k=1, filename="best_train_miou_hsi")
checkpoint_callback_last_epoch = ModelCheckpoint(monitor="epoch", mode="max", save_top_k=1, filename="last_epoch_hsi")

# Set the float32 matmul precision to 'medium' or 'high'
torch.set_float32_matmul_precision('medium')

trainer = L.Trainer(
    max_epochs=max_num_epochs, 
    accumulate_grad_batches=accumulate_grad_batches, 
    callbacks=[
        EarlyStopping(monitor="train_loss", mode="min", verbose=True, patience=10), 
        checkpoint_callback_train_loss, checkpoint_callback_last_epoch ,  checkpoint_callback_train_miou, StochasticWeightAveraging(swa_lrs=swa_lr) ], 
    accelerator="gpu", 
    devices="auto", 
    gradient_clip_val=grad_clip_val, 
    precision="16-mixed" ) 

if training_model == True: 
    
    if tuning_model:
        tuner = Tuner(trainer)

        # batch_finder = tuner.scale_batch_size(model, mode="binsearch")

        # below can be used to find the lr_for the model
        lr_finder = tuner.lr_find(model)

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()

        # update hparams of the model
        model.hparams.learning_rate = new_lr  # learning_rate
        model.hparams.batch_size = batch_size # batch_finder

        print("learning rate:", model.hparams.learning_rate, "batch size:", model.hparams.batch_size)

        hparams = model.hparams
    else:
        model.hparams.learning_rate = initial_lr  # learning_rate
        model.hparams.batch_size = batch_size
        
    trainer.fit(model)

if test_model:
    
    # need to load the model again to get the best model
    # process the test set and save the results to folder 
    # make sure to resize the images to the original size before saving the results (256x256)
    # save the predicted results as tiff files without compression and have the same name as the original image

    model = SMP_SemanticSegmentation.load_from_checkpoint("lightning_logs/version_38/checkpoints/lowest_train_loss_hsi.ckpt")
    test_dataset = mmsegyrebDataset(image_set="test", root_dir=dataset_dir,  transform=test_transform)
    results_dir = "submitted_results/submission1/results"
    model.results_dir = results_dir

    os.makedirs(results_dir, exist_ok=True)
    test_loader = DataLoader( test_dataset, shuffle=False, collate_fn=models.collate_fn,num_workers=16, batch_size =16)
    model.eval()

    trainer.test(model, dataloaders=test_loader)

    print("Test set processed and results saved.")
    # you can now zip the folder with zip_files.py file