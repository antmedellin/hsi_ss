import torch, torch.nn as nn, torch.utils.data as data, torchvision as tv, torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix, MulticlassF1Score
from torchmetrics.segmentation import MeanIoU
import torch.optim.lr_scheduler as lr_scheduler 
import matplotlib
# matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt  
import seaborn as sns
import pandas as pd
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import JaccardLoss
import math
import random
import warnings
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from segmentation_models_pytorch.losses import FocalLoss, LovaszLoss, DiceLoss, SoftCrossEntropyLoss
from torchvision.transforms import Resize
import tifffile as tiff
import os
import sys



class CombinedLoss(nn.Module):
    def __init__(self, ignore_index=0):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss =  nn.CrossEntropyLoss(ignore_index=ignore_index)
        # self.focal_loss = FocalLoss(mode="multiclass", ignore_index=ignore_index)
        # self.JaccardLoss = JaccardLoss(mode="multiclass") # focuses on the iou metric, range 0-1
        self.LovaszLoss = LovaszLoss(mode="multiclass", ignore_index=ignore_index) # focuses on the iou metric, range 0-1
        self.DiceLoss = DiceLoss(mode="multiclass", ignore_index=ignore_index) # focuses on the iou metric, range 0-1
        # add f1 score based loss function 
        # https://www.kaggle.com/code/rejpalcz/best-loss-function-for-f1-score-metric 
        # https://smp.readthedocs.io/en/latest/losses.html 
     

    def forward(self, logits, targets):
        # focal_loss = self.focal_loss(logits, targets)
        # jaccard_loss = self.JaccardLoss(logits, targets)
        lovasz_loss = self.LovaszLoss(logits, targets)
        dice_loss = self.DiceLoss(logits, targets)
        ce_loss = self.cross_entropy_loss(logits, targets)
        return   0.5* ce_loss +  dice_loss + 3 * lovasz_loss # scale iou loss since it is smaller than focal loss

def collate_fn(inputs):

    batch = dict()
    batch["msi_pixel_values"] = torch.stack([i[0] for i in inputs], dim=0)
    batch["sar_pixel_values"] = torch.stack([i[1] for i in inputs], dim=0)
    batch["labels"] = torch.stack([i[2] for i in inputs], dim=0).long()
    batch["filenames"] = [i[3] for i in inputs]

    return batch   
        
class BaseSegmentationModel(L.LightningModule):
        def __init__(self, num_classes, learning_rate = 1e-3, ignore_index=0 ,num_channels=12, num_workers=4, train_dataset=None, val_dataset=None, test_dataset = None, batch_size=2, results_dir="results" ):
            super().__init__()
            
            self.learning_rate = learning_rate
            # self.batch_size = batch_size override in dataloaders
            self.ignore_index = ignore_index
            self.num_workers = num_workers
            self.num_classes = num_classes
            self.num_channels = num_channels
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.test_dataset = test_dataset
            self.results_dir = results_dir
            
            self.save_hyperparameters()
            
            self.loss_fn = CombinedLoss(ignore_index=self.ignore_index)
            
            self.train_miou = MeanIoU(num_classes=self.num_classes, per_class=False)
            self.test_miou = MeanIoU(num_classes=self.num_classes, per_class=False)
            self.val_miou = MeanIoU(num_classes=self.num_classes, per_class=False)
            
            self.train_confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes, normalize="true", ignore_index=self.ignore_index)
            self.val_confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes, normalize="true", ignore_index=self.ignore_index)
            self.test_confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes, normalize="true", ignore_index=self.ignore_index)
            
            #  Calculate statistics for each label and average them
            self.train_acc_mean = MulticlassAccuracy(num_classes=self.num_classes, average="macro", ignore_index=self.ignore_index)
            self.val_acc_mean = MulticlassAccuracy(num_classes=self.num_classes, average="macro", ignore_index=self.ignore_index)
            self.test_acc_mean = MulticlassAccuracy(num_classes=self.num_classes, average="macro", ignore_index=self.ignore_index)
            
            #  Sum statistics over all labels
            self.train_acc_overall = MulticlassAccuracy(num_classes=self.num_classes, average="micro", ignore_index=self.ignore_index)
            self.val_acc_overall = MulticlassAccuracy(num_classes=self.num_classes, average="micro", ignore_index=self.ignore_index)
            self.test_acc_overall = MulticlassAccuracy(num_classes=self.num_classes, average="micro", ignore_index=self.ignore_index)
        
            # Mean F1 Score
            self.train_f1_mean = MulticlassF1Score(num_classes=self.num_classes, average="macro", ignore_index=self.ignore_index)
            self.val_f1_mean = MulticlassF1Score(num_classes=self.num_classes, average="macro", ignore_index=self.ignore_index)
            self.test_f1_mean = MulticlassF1Score(num_classes=self.num_classes, average="macro", ignore_index=self.ignore_index)


        def forward(self, msi_img, sar_img):
            raise NotImplementedError("Subclasses should implement this method")
        
        def log_cf(self, result_cf, step_type):
            
            confusion_matrix_computed = result_cf.detach().cpu().numpy()
            df_cm = pd.DataFrame(confusion_matrix_computed)
            plt.figure(figsize = (self.num_classes+5,self.num_classes))
            fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
            plt.close(fig_)
            self.loggers[0].experiment.add_figure(f"Confusion Matrix {step_type}", fig_, self.current_epoch)
        
        def log_data(self, step_type, logits, labels, loss):
            
            preds = torch.argmax(logits, dim=1)
            
            # Check the shapes of preds and labels
            # print(f"Shape of preds: {preds.shape}, dtype: {preds.dtype}")
            # print(f"Shape of labels: {labels.shape}, dtype: {labels.dtype}")
            
            assert preds.shape == labels.shape, "Predictions and labels must have the same shape"
            # Check for NaNs or Infs
            if torch.isnan(preds).any() or torch.isinf(preds).any():
                raise ValueError("preds contain NaNs or Infs")
            if torch.isnan(labels).any() or torch.isinf(labels).any():
                raise ValueError("labels contain NaNs or Infs")
            
            # Check unique values
            # print(f"Unique values in preds: {torch.unique(preds)}")
            # print(f"Unique values in labels: {torch.unique(labels)}")

            # Check number of classes
            # num_classes_preds = len(torch.unique(preds))
            # num_classes_labels = len(torch.unique(labels))
            # print(f"Number of classes in preds: {num_classes_preds}")
            # print(f"Number of classes in labels: {num_classes_labels}")
            
            # Ensure preds has the correct number of classes
            # if num_classes_preds != self.train_miou.num_classes:
            #     raise ValueError(f"Number of classes in preds ({num_classes_preds}) does not match expected ({self.train_miou.num_classes})")

    
            
            if step_type == "train":
                # result_cf = self.train_confusion_matrix(preds, labels) # not used in training loop
                result_miou = self.train_miou(preds, labels)
                result_acc_overall = self.train_acc_overall(preds, labels)
                results_acc_mean = self.train_acc_mean(preds, labels)
                results_f1_mean = self.train_f1_mean(preds, labels)
                # print("train", result_miou, result_acc_overall, results_acc_mean)
                optimizer = self.optimizers()
                lr = optimizer.param_groups[0]['lr']
                self.log(f"{step_type}_learning_rate", lr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            elif step_type == "val":
                # result_cf = self.val_confusion_matrix(preds, labels)
                result_miou = self.val_miou(preds, labels)
                result_acc_overall = self.val_acc_overall(preds, labels)
                results_acc_mean = self.val_acc_mean(preds, labels)
                results_f1_mean = self.val_f1_mean(preds, labels)
                # self.log_cf(result_cf, step_type)
            elif step_type == "test":
                # result_cf = self.test_confusion_matrix(preds, labels)
                result_miou = self.test_miou(preds, labels)
                result_acc_overall = self.test_acc_overall(preds, labels)
                results_acc_mean = self.test_acc_mean(preds, labels)
                results_f1_mean = self.test_f1_mean(preds, labels)
                # self.log_cf(result_cf, step_type)
            else:
                raise ValueError("step_type must be one of 'train', 'val', or 'test'")
            
            self.log(f"{step_type}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{step_type}_accuracy_overall", result_acc_overall, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{step_type}_accuracy_mean", results_acc_mean, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{step_type}_miou", result_miou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{step_type}_f1_mean", results_f1_mean, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            
           
        def training_step(self, batch):
            
            step_type = "train"
            sar_pixel_values = batch["sar_pixel_values"]
            msi_pixel_values = batch["msi_pixel_values"]
            labels = batch["labels"]     
            
            logits = self.forward(msi_pixel_values,sar_pixel_values)
            loss = self.loss_fn(logits, labels) 
            
            self.log_data(step_type, logits, labels, loss)

            return loss
        
        def test_step(self, batch):
            
            step_type = "test"
            sar_pixel_values = batch["sar_pixel_values"]
            msi_pixel_values = batch["msi_pixel_values"]
            labels = batch["labels"]    
            filenames = batch["filenames"] 
            
            
            logits = self.forward(msi_pixel_values,sar_pixel_values)
            
            # Assuming outputs are logits, apply softmax to get probabilities
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Resize transform to resize predictions to 256x256
            resize_transform = Resize((256, 256))
            
            for i, prediction in enumerate(predictions):
                # Resize the prediction to the original size (256x256)
                resized_prediction = resize_transform(torch.tensor(prediction).unsqueeze(0)).squeeze(0).numpy()
                
                # # view the prediction 
                # plt.imshow(resized_prediction)
                # plt.show()
        
                # Save the prediction as a TIFF file without compression
                result_path = os.path.join(self.results_dir, filenames[i])
                tiff.imwrite(result_path, resized_prediction.astype(np.uint8), compression=None)

            
    

            return logits
        
        # def validation_step(self, batch):
                
        #     step_type = "val"
        #     sar_pixel_values = batch["sar_pixel_values"]
        #     msi_pixel_values = batch["msi_pixel_values"]
        #     labels = batch["labels"]     
            
        #     logits = self.forward(msi_pixel_values,sar_pixel_values)
        #     loss = self.loss_fn(logits, labels) 
            
        #     self.log_data(step_type, logits, labels, loss)

        #     return loss
        
        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
            # return optimizer
            # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

            return {
            'optimizer': optimizer,
            # "lr_scheduler": scheduler
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_miou',  # Metric to monitor for learning rate adjustment
                'interval': 'epoch',    # How often to apply the scheduler
                'frequency': 1          # Frequency of the scheduler
            }
             }     
              
        def train_dataloader(self):
            
            return  DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, collate_fn=collate_fn,num_workers=self.num_workers)
        
        # def val_dataloader(self):
            
        #     return  DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, collate_fn=collate_fn,num_workers=self.num_workers)
        
        def test_dataloader(self):
            
            return  DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, collate_fn=collate_fn,num_workers=self.num_workers)

class SMP_SemanticSegmentation(BaseSegmentationModel):
    def __init__(self, num_classes, learning_rate=1e-3, ignore_index=0, num_channels=12, num_workers=4, train_dataset=None, val_dataset=None, test_dataset=None, batch_size=2):
        super().__init__(num_classes, learning_rate, ignore_index, num_channels, num_workers, train_dataset, val_dataset, test_dataset, batch_size)
        
        # can replace with models from segmentation_models_pytorch
        # refernce: https://segmentation-modelspytorch.readthedocs.io/en/latest/#models 
        self.msi_unet = smp.FPN('resnet152', in_channels=self.num_channels, classes=self.num_classes, encoder_depth=5)
        # self.hsi_unet = smp.PSPNet('resnet152', in_channels=self.num_channels, classes=self.num_classes, encoder_depth=5, upsampling=32)
        # self.hsi_unet = smp.PAN('resnet152', in_channels=self.num_channels, classes=self.num_classes) # batch size 16 
        #Unet, Linknet, FPN, PSPNet, PAN

    def forward(self, msi_img, sar_img):
            
            x = self.msi_unet(msi_img)
            
            return x
        
 
class DINOv2_SemanticSegmentation(BaseSegmentationModel):
    def __init__(self, num_classes, learning_rate=1e-3, ignore_index=0, num_channels=204, num_workers=4, train_dataset=None, val_dataset=None, test_dataset=None, batch_size=2, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14_reg", half_precision=False , tokenW=32, tokenH=32 ):
        super().__init__(num_classes, learning_rate, ignore_index, num_channels, num_workers, train_dataset, val_dataset, test_dataset, batch_size)
        
        # load the dinov2 model 
        if half_precision:
            self.dinov2 = torch.hub.load(repo_or_dir=repo_name, model=model_name).half().to(self.device)
        else:
            self.dinov2= torch.hub.load(repo_or_dir=repo_name, model=model_name).to(self.device)
            
        last_layer_params = list(self.dinov2.parameters())[-1]
        patch_descriptor_size = last_layer_params.shape[0]
        
        
        # Freeze the DINOv2 model. This allows for faster training. 
        for _, param in self.dinov2.named_parameters():
            param.requires_grad = False
        
        self.classifier = torch.nn.Conv2d(patch_descriptor_size, num_classes, (1,1))
        
        
        self.patch_descriptor_size = patch_descriptor_size
        self.tokenW = tokenW
        self.tokenH = tokenH

    def forward(self, msi_img, sar_img):
            
            # assert not torch.isnan(rgb_pixel_values).any(), "NaN values in input pixel_values"            
            embeddings = self.dinov2.get_intermediate_layers(sar_img)[0].squeeze()
            
            assert not torch.isnan(embeddings).any(), "NaN values in embeddings"
            
            embeddings = embeddings.reshape(-1, self.tokenW, self.tokenH, self.patch_descriptor_size)
            embeddings = embeddings.permute(0,3,1,2)
        
            # assert not torch.isnan(embeddings).any(), "NaN values in embeddings"
            
            logits = self.classifier(embeddings)
            # print( logits[0])
            assert not torch.isnan(logits).any(), "NaN values in logits"
            logits = torch.nn.functional.interpolate(logits, size=sar_img.shape[2:], mode="bilinear", align_corners=False)
            
            return logits     

