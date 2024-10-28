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
from models import SMP_SemanticSegmentation, DINOv2_SemanticSegmentation, SMP_Channel_SemanticSegmentation, VisionTransformer, DiffVisionTransformer, SWINTransformer,Pretrained_models, Pretrained_backbone_uperhead
# import tifffile as tiff
from torchvision.transforms import Resize

# tensorboard --logdir=./lightning_logs/
# ctrl shft p -> Python: Launch Tensorboard  select lightning logs

   
def extract_rgb(cube, red_layer=70 , green_layer=53, blue_layer=19):

    red_img = cube[ red_layer,:,:]
    green_img = cube[ green_layer,:,:]
    blue_img = cube[ blue_layer,:,:]
        
    data=np.stack([red_img,green_img,blue_img], axis=-1)
    
    # convert from x,y,channels to channels, x, y
    # data = np.transpose(data, (2, 0, 1))
    
    return data 

def GDAL_imreadmulti(file_name):
    # Open the dataset
    dataset = gdal.Open(file_name)

    # Check if opened
    if dataset:
      # print("Dataset opened...")
      width = dataset.RasterXSize
      height = dataset.RasterYSize
      num_bands = dataset.RasterCount

      image_bands = []

      for band_num in range(1, num_bands+1):
        band = dataset.GetRasterBand(band_num)

        # Read band data
        band_data = band.ReadAsArray()

        # Create an OpenCV Mat from the band data
        band_mat = np.array(band_data, dtype='float32')

        # Correct the orientation of the image
        band_mat = np.transpose(band_mat)
        band_mat = cv2.flip(band_mat, 1)
        
        # Normalize the band data to the range [0, 1]
        band_min = band_mat.min()
        band_max = band_mat.max()
        normalized_band_mat = (band_mat - band_min) / (band_max - band_min)
        band_mat = normalized_band_mat

        # Apply threshold and convert to 8-bit unsigned integers
        _, band_mat = cv2.threshold(band_mat, 1.0, 1.0, cv2.THRESH_TRUNC)
        band_mat = cv2.convertScaleAbs(band_mat, alpha=(255.0))

        # Add the processed band to the list
        image_bands.append(band_mat)
        
        cube=np.array(image_bands)
        
        # convert from x,y,channels to channels, x, y
        # data = np.transpose(cube, (2, 0, 1))
        
      return True, cube

    else:
      print("GDAL Error: ", gdal.GetLastErrorMsg())
      return False, []

class LIBHSIDataset(Dataset):
    def __init__(self, image_set,  root_dir, id2color, transform=None):
        
        # image_set # train ,test, validation
        self.transform = transform
        self.root = join(root_dir,image_set)
        
        # Convert id2color to a numpy array for easier comparison
        self.id2color_np = np.array(list(id2color.values()))

        self.img_dir =  join(self.root, "reflectance_cubes")
        self.label_dir = join(self.root, "labels")
        
        self.img_names = [f for f in os.listdir(self.img_dir) if f.endswith('.' + 'dat')]
        self.num_images = len( self.img_names  ) 

        # print("Number of images in the dataset: ", self.num_images, "label_dir len: ", len(os.listdir(self.label_dir)))
        assert self.num_images == len(os.listdir(self.label_dir))
        
        self.img_labels = [f for f in os.listdir(self.label_dir)]
        
        # sort img_names and img_labels
        self.img_names.sort()
        self.img_labels.sort()

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        label_name, ext_label = os.path.splitext(self.img_labels[idx])
        
        hsi_name, ext_hsi = os.path.splitext(self.img_names[idx])
        
        assert label_name == hsi_name # make sure they have the same name 
        
        # read the label image 
        label_path = join(self.label_dir, self.img_labels[idx])
        label_img = Image.open(label_path).convert('RGB')
        
        label_img_np = np.array(label_img) # uint8 x,y,channels
        
        #convert labeled rgb image to greyscale
        label_img_greyscale = np.zeros(label_img_np.shape[:2], dtype=np.uint8)
        for i, color in enumerate(self.id2color_np):
            # Find where in the target the current color is
            mask = np.all(label_img_np == color, axis=-1)
            
            # Wherever the color is found, set the corresponding index in target_new to the current class label
            label_img_greyscale[mask] = i
        
        hsi_path = join(self.img_dir, self.img_names[idx])
        _, hsi_img = GDAL_imreadmulti(hsi_path)
   
        rgb_img = extract_rgb(hsi_img) 

        hsi_img = np.transpose(hsi_img, (1, 2, 0)) # transpose to x,y,channels for albumnetations
        
        # apply transformations  # must be in x,y,channels format        
        if self.transform:            
            transformed = self.transform(image = rgb_img, mask = label_img_greyscale, hsi_image = hsi_img)
        
            hsi_img, rgb_img, label_img_greyscale = torch.tensor(transformed['hsi_image']), torch.tensor(transformed['image']), torch.tensor(transformed['mask'])
        else:
            hsi_img, rgb_img, label_img_greyscale = torch.tensor(hsi_img), torch.tensor(rgb_img), torch.tensor(label_img_greyscale)
            
            
        #convert from x,y,channels to channels, x, y
        hsi_img = hsi_img.permute(2,0,1)
        rgb_img = rgb_img.permute(2,0,1)
        
        #convert from uint8 to float32
        hsi_img = hsi_img.float()
        rgb_img = rgb_img.float()
            
        return hsi_img, rgb_img, label_img_greyscale   
    
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
        
        # # Add a third channel of zeros to the SAR image # used in cases where 3 channel rgb models are used
        # zeros_channel = np.zeros((sar_img_np.shape[0], sar_img_np.shape[1]), dtype=sar_img_np.dtype)
        # sar_img_np = np.dstack((sar_img_np, zeros_channel)) # Now sar_img_np has shape (x, y, 3)
        
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


def compute_dataset_statistics(dataset, num_channels):
    # Initialize accumulators
    channel_sum = torch.zeros(num_channels)
    channel_sum_squared = torch.zeros(num_channels)
    channel_min = torch.full((num_channels,), float('inf'))
    channel_max = torch.full((num_channels,), float('-inf'))
    pixel_count = 0

    # Create DataLoader
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=models.collate_fn,num_workers=8)

    for batch in dataloader:
        images = batch['hsi_pixel_values']  # Assuming the dataset returns a dictionary with 'image' key
        images = images.view(images.size(0), images.size(1), -1)  # Flatten the spatial dimensions
        pixel_count += images.size(2)

        # Update accumulators
        channel_sum += images.sum(dim=2).sum(dim=0)
        channel_sum_squared += (images ** 2).sum(dim=2).sum(dim=0)
        channel_min = torch.min(channel_min, images.min(dim=2)[0].min(dim=0)[0])
        channel_max = torch.max(channel_max, images.max(dim=2)[0].max(dim=0)[0])

    # Compute mean and std
    channel_mean = channel_sum / pixel_count
    channel_std = torch.sqrt(channel_sum_squared / pixel_count - channel_mean ** 2)

    return channel_min, channel_max, channel_mean, channel_std
    

# dataset_dir='/workspaces/MMSeg-YREB'

dataset_dir='/workspaces/LIB-HSI'
rgb_data_json = '/workspaces/hsi_ss/lib_hsi_rgb.json'
file_data =  open(rgb_data_json)
file_contents = json.load(file_data)
id2label ={}
id2color = {}
for i, item in enumerate(file_contents['items'], start=0):
    id2label[i] = item['name']
    id2color[i] = [item['red_value'], item['green_value'], item['blue_value']]
# print(id2label)
# print(id2color)
num_classes = len(id2label)


# 9 LULC classes are: 0) Background, 1) Tree, 2) Grassland, 3) Cropland, 4) Low Vegetation, 5) Wetland, 6) Water, 7) Built-up, 8) Bare ground, 9) Snow.
# num_classes = 10 # ignore 0 background 
num_classes = len(id2label)

batch_size = 4
# ignore_index=0 # background
ignore_index=7 # misc. class, 

num_workers = 4 #  os.cpu_count() or 1  # Fallback to 1 if os.cpu_count() is None
initial_lr =  1e-3  # .001 for smp, 3e-4 for transformer
swa_lr = 0.01
# these should be multiple of 14 for dino model 
# input image is of size 256x256
img_height = 512  #512
img_width = 512  #256
max_num_epochs = 1000
accumulate_grad_batches = 8 # increases the effective batch size  # 1 means no accumulation # more important when batch size is small or not doing multi gpu training
grad_clip_val = 5 # clip gradients that have norm bigger than tmax_val)his
training_model = True
tuning_model = False
test_model = False
# min_epochs = 20

# Define mean and standard deviation for normalization
# Use the same value for all channels
# num_channels = 14
num_channels = 204
# mean = [0.45] * num_channels  
# std = [0.225] * num_channels   

pretrained_mean = [0.485, 0.456, 0.406]
pretrained_std = [0.229, 0.224, 0.225]
img_height = 224
img_width = 224

torch.cuda.empty_cache()

test_transform = A.Compose([
    A.Resize(width=img_width, height=img_height), 
    # A.Normalize(normalization="image", max_pixel_value=255.0)
    A.Normalize(mean=pretrained_mean, std=pretrained_std, max_pixel_value=255.0),
], additional_targets={"hsi_image": "image"})

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, p=0.5),  # Set alpha_affine to None
    A.Resize(width=img_width, height=img_height), 
    # A.Normalize(normalization="image", max_pixel_value=255.0),
    A.Normalize(mean=pretrained_mean, std=pretrained_std, max_pixel_value=255.0),
    A.ChannelDropout(channel_drop_range=(1, 2), fill_value=0, p=0.5)
], additional_targets={"hsi_image": "image"})


# mean = [1368.6125, 1159.3759, 1066.2174, 1000.4777, 1233.8301, 1868.6222, 2124.8076, 2111.7993, 2322.8552, 1078.5980, 1715.7476, 1081.0111, -1487.6545, -803.5430]
# std = [491.7540, 544.0209, 557.4269, 675.8509, 660.5626, 602.1366, 635.2303, 644.9349, 678.6457, 557.9268, 665.9048, 528.2560, 430.0971, 362.5961]

# min_val = [688., 467., 294., 173., 173., 196., 188., 177., 181., 82., 61., 30., -7992., -7478.]
# max_val = [11824., 13134., 13005., 14074., 13930., 13746., 14287., 15032., 14416., 9006., 11881., 11510., 5699., 6385.]


# Channel Min: tensor([  688.,   467.,   294.,   173.,   173.,   196.,   188.,   177.,   181.,
#            82.,    61.,    30., -7992., -7478.])

# Channel Max: tensor([11824., 13134., 13005., 14074., 13930., 13746., 14287., 15032., 14416.,
#          9006., 11881., 11510.,  5699.,  6385.])

# Channel Mean: tensor([ 1368.6125,  1159.3759,  1066.2174,  1000.4777,  1233.8301,  1868.6222,
#          2124.8076,  2111.7993,  2322.8552,  1078.5980,  1715.7476,  1081.0111,
#         -1487.6545,  -803.5430])

# Channel Std: tensor([491.7540, 544.0209, 557.4269, 675.8509, 660.5626, 602.1366, 635.2303,
#         644.9349, 678.6457, 557.9268, 665.9048, 528.2560, 430.0971, 362.5961])


# train_dataset = mmsegyrebDataset(image_set="train", root_dir=dataset_dir,  transform=train_transform)

train_dataset = LIBHSIDataset(image_set="train", root_dir=dataset_dir, id2color=id2color, transform=train_transform)
test_dataset = LIBHSIDataset(image_set="test", root_dir=dataset_dir, id2color=id2color,  transform=test_transform)
val_dataset = LIBHSIDataset(image_set="validation", root_dir=dataset_dir, id2color=id2color, transform=test_transform)


# channel_min, channel_max, channel_mean, channel_std = compute_dataset_statistics(test_dataset, num_channels)

# print("Channel Min:", channel_min)
# print("Channel Max:", channel_max)
# print("Channel Mean:", channel_mean)
# print("Channel Std:", channel_std)
# sys.exit()

# model = SMP_SemanticSegmentation(num_classes=num_classes,learning_rate=initial_lr, ignore_index=ignore_index, num_channels= num_channels, num_workers=num_workers,  train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, batch_size=batch_size)


# model = DINOv2_SemanticSegmentation(num_classes=num_classes,learning_rate=initial_lr, ignore_index=ignore_index, num_channels= num_channels, num_workers=num_workers,  train_dataset=train_dataset,  batch_size=batch_size)
# model = SMP_Channel_SemanticSegmentation(num_classes=num_classes,learning_rate=initial_lr, ignore_index=ignore_index, num_channels= num_channels, num_workers=num_workers,  train_dataset=train_dataset, batch_size=batch_size)
# model = SMP_SemanticSegmentation.load_from_checkpoint("lightning_logs/version_19/checkpoints/lowest_train_loss_hsi.ckpt")

# model = VisionTransformer(num_classes=num_classes,learning_rate=initial_lr, ignore_index=ignore_index, num_channels= num_channels, num_workers=num_workers,  train_dataset=train_dataset,val_dataset=val_dataset, test_dataset=test_dataset, batch_size=batch_size, image_size=img_height)


# model = VisionTransformer.load_from_checkpoint("lightning_logs/version_39/checkpoints/lowest_train_loss_hsi.ckpt")

# model = DiffVisionTransformer(num_classes=num_classes,learning_rate=initial_lr, ignore_index=ignore_index, num_channels= num_channels, num_workers=num_workers,  train_dataset=train_dataset, batch_size=batch_size)

# model = SWINTransformer(num_classes=num_classes,learning_rate=initial_lr, ignore_index=ignore_index, num_channels= num_channels, num_workers=num_workers,  train_dataset=train_dataset,val_dataset=val_dataset, test_dataset=test_dataset, batch_size=batch_size, image_size=img_height)

# model = Pretrained_models(num_classes=num_classes,learning_rate=initial_lr, ignore_index=ignore_index, num_channels= num_channels, num_workers=num_workers,  train_dataset=train_dataset,val_dataset=val_dataset, test_dataset=test_dataset, batch_size=batch_size, image_size=img_height)


model = Pretrained_backbone_uperhead(num_classes=num_classes,learning_rate=initial_lr, ignore_index=ignore_index, num_channels= num_channels, num_workers=num_workers,  train_dataset=train_dataset,val_dataset=val_dataset, test_dataset=test_dataset, batch_size=batch_size, image_size=img_height)


# sudo apt-get install graphviz -y
# pip install torchviz
# from torchviz import make_dot  
# Create a sample input tensor with the appropriate shape
# Adjust the shape according to your model's expected input
sample_msi_img = torch.randn(8, num_channels, img_height, img_height)  # Example shape
sample_rgb_img = torch.randn(8, 3, img_height, img_height)  # Example shape for RGB image


# print(train_dataset[0])
# sample_msi_img = train_dataset[0][0].unsqueeze(0)
# sample_rgb_img = train_dataset[0][1].unsqueeze(0)

# # Pass the sample input through the model
output = model.forward(sample_msi_img, sample_rgb_img)

# # Visualize the computation graph
# graph = make_dot(output, params=dict(model.named_parameters()))
# graph.render("model_computation_graph", format="png")

# sys.exit()

checkpoint_callback_val_loss = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="lowest_val_loss_hsi")
# checkpoint_callback_train_miou = ModelCheckpoint(monitor="train_miou", mode="max", save_top_k=1, filename="best_train_miou_hsi")
# checkpoint_callback_last_epoch = ModelCheckpoint(monitor="epoch", mode="max", save_top_k=1, filename="last_epoch_hsi")

# Set the float32 matmul precision to 'medium' or 'high'
torch.set_float32_matmul_precision('medium')

trainer = L.Trainer(
    max_epochs=max_num_epochs, 
    accumulate_grad_batches=accumulate_grad_batches, 
    callbacks=[
        EarlyStopping(monitor="val_loss", mode="min", verbose=True, patience=15), 
        checkpoint_callback_val_loss, StochasticWeightAveraging(swa_lrs=swa_lr) ], 
    accelerator="gpu", 
    devices="auto", 
    gradient_clip_val=grad_clip_val, 
    precision="16-mixed" ) # 

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

    model = Pretrained_backbone_uperhead.load_from_checkpoint("lightning_logs/version_112/checkpoints/lowest_val_loss_hsi.ckpt")

    model.eval()

    trainer.test(model)

    print("Done")
    