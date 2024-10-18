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
from crfrnn import CrfRnn

from scipy.ndimage import distance_transform_edt
from rotary import RotaryEmbedding
from rotary import apply_rotary_emb
from local_attention import LocalAttention
from flash_attn import flash_attn_func
from transformers import Swinv2Config, Swinv2Model, UperNetConfig, UperNetForSemanticSegmentation
from transformers import ConvNextConfig, Swinv2Config, ConvNextV2Config, SwinConfig
#  pip install local-attention
# pip install flash-attn --no-build-isolation
# https://github.com/xiayuqing0622/customized-flash-attention 

class BoundaryLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(BoundaryLoss, self).__init__()
        self.weight = weight

    def forward(self, logits, targets):
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)
        
        # Get the predicted class for each pixel
        preds = torch.argmax(probs, dim=1)
        
        # Compute the distance transform for the ground truth
        dist_maps = self.compute_distance_transform(targets)
        
        # Compute the boundary loss
        boundary_loss = torch.mean(dist_maps * (preds != targets).float())
        
        return self.weight * boundary_loss

    def compute_distance_transform(self, targets):
        dist_maps = torch.zeros_like(targets, dtype=torch.float32)
        for b in range(targets.shape[0]):
            for c in range(targets.shape[1]):
                target = targets[b, c].cpu().numpy()
                dist_map = distance_transform_edt(target == 0)
                dist_maps[b, c] = torch.tensor(dist_map, device=targets.device)
        return dist_maps

class CombinedLoss(nn.Module):
    def __init__(self, ignore_index=0):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss =  nn.CrossEntropyLoss(ignore_index=ignore_index)
        # self.focal_loss = FocalLoss(mode="multiclass", ignore_index=ignore_index)
        self.JaccardLoss = JaccardLoss(mode="multiclass") # focuses on the iou metric, range 0-1
        self.LovaszLoss = LovaszLoss(mode="multiclass", ignore_index=ignore_index) # focuses on the iou metric, range 0-1
        self.DiceLoss = DiceLoss(mode="multiclass", ignore_index=ignore_index) # focuses on the iou metric, range 0-1
        # add f1 score based loss function 
        # https://www.kaggle.com/code/rejpalcz/best-loss-function-for-f1-score-metric 
        # https://smp.readthedocs.io/en/latest/losses.html 
        # self.BoundaryLoss = BoundaryLoss(weight=1.0)


    def forward(self, logits, targets):
        # focal_loss = self.focal_loss(logits, targets)
        jaccard_loss = self.JaccardLoss(logits, targets)
        lovasz_loss = self.LovaszLoss(logits, targets)
        dice_loss = self.DiceLoss(logits, targets)
        ce_loss = self.cross_entropy_loss(logits, targets)
        # boundary_loss = self.BoundaryLoss(logits, targets)

        return   1 * ce_loss +   2 * dice_loss + 3 * lovasz_loss + 3 * jaccard_loss #+ 1 * boundary_loss
        
        # scale iou loss since it is smaller than focal loss

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
            # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
            
            # Learning rate warmup scheduler for a warmup period of 5 epochs
            def lr_lambda(epoch):
                if epoch < 5:
                    return float(epoch) / 5
                return 1.0

            # warmup_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
            
            # Cosine annealing warm restarts scheduler
            # cosine_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)
            
            # Combine the warmup and cosine annealing schedulers
            # scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5])
        

            return {
            'optimizer': optimizer,
            # "lr_scheduler": scheduler
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',  # Metric to monitor for learning rate adjustment
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


class SMP_Channel_SemanticSegmentation(BaseSegmentationModel):
    def __init__(self, num_classes, learning_rate=1e-3, ignore_index=0, num_channels=12, num_workers=4, train_dataset=None, val_dataset=None, test_dataset=None, batch_size=2):
        super().__init__(num_classes, learning_rate, ignore_index, num_channels, num_workers, train_dataset, val_dataset, test_dataset, batch_size)
        
        # can replace with models from segmentation_models_pytorch
        # refernce: https://segmentation-modelspytorch.readthedocs.io/en/latest/#models 
        # self.msi_smp = smp.FPN('resnet152', in_channels=self.num_channels, classes=self.num_classes, encoder_depth=5)
        # self.hsi_unet = smp.PSPNet('resnet152', in_channels=self.num_channels, classes=self.num_classes, encoder_depth=5, upsampling=32)
        # self.hsi_unet = smp.PAN('resnet152', in_channels=self.num_channels, classes=self.num_classes) # batch size 16 
        #Unet, Linknet, FPN, PSPNet, PAN
        
        # Initialize separate  models for each channel
        self.fpn_models = nn.ModuleList([
            smp.FPN('resnet34', in_channels=1, classes=self.num_classes, encoder_depth=5)
            for _ in range(self.num_channels)
        ])
        
        # Learnable fusion layer
        self.fusion_layer = nn.Conv2d(self.num_channels * self.num_classes, self.num_classes, kernel_size=1)


    def forward(self, msi_img, sar_img):
            
        # x = self.msi_smp(msi_img)
            
        # return x        
        # Split the input image into separate channels
        channel_outputs = []
        for i in range(self.num_channels):
            channel_img = msi_img[:, i:i+1, :, :]  # Extract the i-th channel
            channel_output = self.fpn_models[i](channel_img)
            channel_outputs.append(channel_output)
        
        # Fuse the results from each channel (e.g., by summing)
        # fused_output = torch.stack(channel_outputs, dim=0).sum(dim=0)
        
        # Concatenate the results from each channel
        concatenated_output = torch.cat(channel_outputs, dim=1)  # Concatenate along the channel dimension
        
        # Apply the learnable fusion layer
        fused_output = self.fusion_layer(concatenated_output)
        
        
        return fused_output
 
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


#reference  https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/11-vision-transformer.html 
def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Args:
        x: Tensor representing the image of shape [B, C, H, W]
        patch_size: Number of pixels per dimension of the patches (integer)
        flatten_channels: If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x

#usage for above
# img_patches = img_to_patch(mages, patch_size=4, flatten_channels=False)

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """Attention Block. pre- layer norm

        Args:
            embed_dim: Dimensionality of input and attention feature vectors
            hidden_dim: Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads: Number of heads to use in the Multi-Head Attention block
            dropout: Amount of dropout to apply in the feed-forward network

        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x

class LocalAttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, window_size, dropout=0.1):
        super(LocalAttentionBlock, self).__init__()
        self.window_size = window_size
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        windows = x.unfold(1, self.window_size, self.window_size).permute(0, 2, 1, 3).contiguous()
        windows = windows.view(-1, self.window_size, C)
        
        attn_output, _ = self.attention(windows, windows, windows)
        attn_output = attn_output.view(B, -1, C)
        
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        
        return x

# referemce https://github.com/microsoft/unilm/blob/master/Diff-Transformer/multihead_diffattn.py 

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'
    
class MultiheadDiffAttn(nn.Module):
    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # num_heads set to half of Transformer's #heads
        self.num_heads = num_heads 
        self.num_kv_heads = 4#num_heads 
        self.n_rep = self.num_heads // self.num_kv_heads
        
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=False)
    
    def forward(
        self,
        x,
        rel_pos,
        attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)

        q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        q = q.transpose(1, 2)
        
        k = torch.repeat_interleave(k.transpose(1, 2), dim=1, repeats=self.n_rep)
        v = torch.repeat_interleave(v.transpose(1, 2), dim=1, repeats=self.n_rep * 2)
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.zeros([tgt_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(q),
                1 + src_len - tgt_len,
            )

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        attn_weights = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=attn_mask, scale=self.scaling)
        every_other_mask = torch.arange(attn_weights.size(1)) % 2 == 0
        attn = attn_weights[:, every_other_mask, :, :] - lambda_full * attn_weights[:, ~every_other_mask, :, :]

        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)

        attn = self.out_proj(attn)
        return attn

    def forward2( #origianal
            self,
            x,
            rel_pos,
            attn_mask=None,
        ):
            bsz, tgt_len, embed_dim = x.size()
            src_len = tgt_len

            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
            k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
            v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)

            q = apply_rotary_emb(q, *rel_pos, interleaved=True)
            k = apply_rotary_emb(k, *rel_pos, interleaved=True)

            offset = src_len - tgt_len
            q = q.transpose(1, 2)
            k = repeat_kv(k.transpose(1, 2), self.n_rep)
            v = repeat_kv(v.transpose(1, 2), self.n_rep)
            q *= self.scaling
            attn_weights = torch.matmul(q, k.transpose(-1, -2))
            if attn_mask is None:
                attn_mask = torch.triu(
                    torch.zeros([tgt_len, src_len])
                    .float()
                    .fill_(float("-inf"))
                    .type_as(attn_weights),
                    1 + offset,
                )
            attn_weights = torch.nan_to_num(attn_weights)
            attn_weights += attn_mask   
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
                attn_weights
            )

            lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
            lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
            lambda_full = lambda_1 - lambda_2 + self.lambda_init
            attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
            attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
            
            attn = torch.matmul(attn_weights, v)
            attn = self.subln(attn)
            attn = attn * (1 - self.lambda_init)
            attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)

            attn = self.out_proj(attn)
            return attn


class MultiheadFlashDiff1(nn.Module):
    """
    (Recommended)
    DiffAttn implemented with FlashAttention, for packages that support different qk/v dimensions
    e.g., our customized-flash-attention (https://aka.ms/flash-diff) and xformers (https://github.com/facebookresearch/xformers)
    """
    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # num_heads set to half of Transformer's #heads
        self.num_heads = num_heads 
        self.num_kv_heads =  4#num_heads 
        self.n_rep = self.num_heads // self.num_kv_heads
        
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=False)
    
    def forward(
        self,
        x,
        rel_pos,
        attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)

        q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        offset = src_len - tgt_len
        q = q.reshape(bsz, tgt_len, self.num_heads, 2, self.head_dim)
        k = k.reshape(bsz, src_len, self.num_kv_heads, 2, self.head_dim)
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1]
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]
        
        print("v shape", v.shape, "batch size", bsz, "seqlen_k", src_len, "num_kv_heads", self.num_kv_heads, "head_dim", self.head_dim)
        
        attn1 = flash_attn_func(q1, k1, v, causal=True)
        attn2 = flash_attn_func(q2, k2, v, causal=True)
        
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn = attn1 - lambda_full * attn2

        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)
        
        attn = self.out_proj(attn)
        return attn
    
    
class DiffVisionTransformer(BaseSegmentationModel):
    def __init__(self, num_classes, learning_rate=1e-3, ignore_index=0, num_channels=12, num_workers=4, train_dataset=None, val_dataset=None, test_dataset=None, batch_size=2, embed_dim=256, num_heads=8, num_layers=8, patch_size=8, dropout=0, num_registers=4):

        
        super().__init__(num_classes, learning_rate, ignore_index, num_channels, num_workers, train_dataset, val_dataset, test_dataset, batch_size)
        
        self.patch_size = patch_size
        self.num_registers = num_registers
        num_patches = (256 // patch_size) ** 2 # 256 is the image size
        self.num_patches = num_patches

        # Layers/Networks
        self.input_layer = nn.Linear(self.num_channels * (patch_size**2), embed_dim)
        
        self.head_dim = embed_dim // (num_heads * 2)
              
        # diff transformer
        # self.transformer = MultiheadDiffAttn(embed_dim, num_layers, num_heads)
        self.transformer = MultiheadFlashDiff1(embed_dim, num_layers, num_heads)
        
        num_new_heads = num_heads * 2
        head_dim = embed_dim // num_new_heads
        self.rotary_emb = RotaryEmbedding(
            head_dim,
            base=10000.0,
            interleaved=True,
            device=self.device,
        )
        self.seq_len = 1 + self.num_patches + self.num_registers
        
        self.segmentation_head = nn.Conv2d(embed_dim, self.num_classes , kernel_size=1)
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + self.num_patches + num_registers , embed_dim))
        self.registers = nn.Parameter(torch.randn(1, num_registers, embed_dim))

    def forward(self, msi_img, sar_img):
        # Preprocess input
        x = img_to_patch(msi_img, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        registers = self.registers.repeat(B, 1, 1)
        x = torch.cat([cls_token, registers, x], dim=1)
        x = x + self.pos_embedding[:, : T + 1 + self.num_registers]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        
        

        self.rotary_emb._update_cos_sin_cache(self.seq_len, device=self.device, dtype=torch.bfloat16)
        rel_pos = (self.rotary_emb._cos_cached, self.rotary_emb._sin_cached)
        
        x = self.transformer(x, rel_pos) # diff transformer
        x = x.transpose(0, 1) # result is batch size, sequence length, embedding size
        # print("x shape", x.shape, x[ 1 + self.num_registers:, :,:].shape)
        
        # Remove CLS token and registers
        x = x[:, 1 + self.num_registers:, :] # batch size, num patches, embedding size   
        
        # Reshape and apply segmentation head
        B, num_patches, embed_dim = x.shape
        height = width = int(num_patches ** 0.5)  # Assuming num_patches is a perfect square

        x = x.permute(0, 2, 1)  # Change shape to [batch_size, embedding_dim, num_patches]
        x = x.view(B, embed_dim, height, width)  # Reshape to [batch_size, embedding_dim, height, width]

        
        # print("x shape prime", x.shape)
        x = self.segmentation_head(x)
        # Upsample to match input image resolution
        x = F.interpolate(x, scale_factor=self.patch_size, mode='bilinear', align_corners=False)

        return x


# refernce https://github.com/yassouali/pytorch-segmentation/blob/master/models/upernet.py 
class PSPModule(nn.Module):
    # In the original inmplementation they use precise RoI pooling 
    # Instead of using adaptative average pooling
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) 
                                                        for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), in_channels, 
                                    kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                        align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y

class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1)
                                    for ft_size in feature_channels[1:]])
        self.smooth_conv =  nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)] 
                                    * (len(feature_channels)-1))
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels)*fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]
        P = [up_and_add(features[i], features[i-1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1]) #P = [P1, P2, P3, P4]
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in P[1:]]

        x = self.conv_fusion(torch.cat((P), dim=1))
        return x
    
        
class VisionTransformer(BaseSegmentationModel):
        
    def __init__(self, num_classes, learning_rate=1e-3, ignore_index=0, num_channels=12, num_workers=4, train_dataset=None, val_dataset=None, test_dataset=None, batch_size=2, embed_dim=512, hidden_dim=1024, num_heads=16, num_layers=8, patch_size=8, dropout=0.2, num_registers=4):

        """Vision Transformer.

        Args:
            embed_dim: Dimensionality of the input feature vectors to the Transformer
            hidden_dim: Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels: Number of channels of the input (3 for RGB)
            num_heads: Number of heads to use in the Multi-Head Attention block
            num_layers: Number of layers to use in the Transformer
            num_classes: Number of classes to predict
            patch_size: Number of pixels that the patches have per dimension
            num_patches: Maximum number of patches an image can have
            dropout: Amount of dropout to apply in the feed-forward network and
                      on the input encoding


            3 ref for classification
            model_kwargs={
                "embed_dim": 256,
                "hidden_dim": 512,
                "num_heads": 8,
                "num_layers": 6,
                "patch_size": 4,
                "num_channels": 3,
                "num_patches": 64,
                "num_classes": 10,
                "dropout": 0.2,
            },
            lr=3e-4,
        """
        super().__init__(num_classes, learning_rate, ignore_index, num_channels, num_workers, train_dataset, val_dataset, test_dataset, batch_size)

        # num_heads = 8
        # embed_dim = 256
        # num_layers = 8
        
        
        self.patch_size = patch_size
        self.num_registers = num_registers
        num_patches = (256 // patch_size) ** 2 # 256 is the image size
        self.num_patches = num_patches

        # Layers/Networks
        self.input_layer = nn.Linear(self.num_channels * (patch_size**2), embed_dim)
        
        
        
        self.head_dim = embed_dim // (num_heads * 2)
        
        
        #traditional transformer
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        
        
        # upernet  
        #backbone is the transformer encoder 
        self.ppn = PSPModule(embed_dim)
        self.FPN = FPN_fuse(feature_channels=[embed_dim, embed_dim, embed_dim, embed_dim], fpn_out=embed_dim)
        self.segmentation_head = nn.Conv2d(embed_dim, self.num_classes , kernel_size=1)
        
        # original segmentation head
        # self.segmentation_head = nn.Conv2d(embed_dim, self.num_classes , kernel_size=1)
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + self.num_patches + num_registers , embed_dim))
        self.registers = nn.Parameter(torch.randn(1, num_registers, embed_dim))

    def forward(self, msi_img, sar_img):
        # Preprocess input
        x = img_to_patch(msi_img, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        registers = self.registers.repeat(B, 1, 1)
        x = torch.cat([cls_token, registers, x], dim=1)
        x = x + self.pos_embedding[:, : T + 1 + self.num_registers]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        
        x = self.transformer(x) # traditional transformer
        
       
        x = x.transpose(0, 1) # result is batch size, sequence length, embedding size
        
        # # Remove CLS token and registers
        x = x[:, 1 + self.num_registers:, :] # batch size, num patches, embedding size
        # # print("x shape", x.shape)
        
        
        # # Reshape and apply segmentation head
        B, num_patches, embed_dim = x.shape
        height = width = int(num_patches ** 0.5)  # Assuming num_patches is a perfect square

        x = x.permute(0, 2, 1)  # Change shape to [batch_size, embedding_dim, num_patches]
        x = x.view(B, embed_dim, height, width)  # Reshape to [batch_size, embedding_dim, height, width]

        
        # Apply PSPModule
        x = self.ppn(x)

        # Apply FPN_fuse
        x = self.FPN([x, x, x, x])  # Assuming the same feature map for simplicity

            
        # print("x shape prime", x.shape)
        
        x = self.segmentation_head(x)      
        
        # Upsample to match input image resolution
        x = F.interpolate(x, scale_factor=self.patch_size, mode='bilinear', align_corners=False)

        return x
    
    
class SWINTransformer(BaseSegmentationModel):
        
    def __init__(self, num_classes, learning_rate=1e-3, ignore_index=0, num_channels=12, num_workers=4, train_dataset=None, val_dataset=None, test_dataset=None, batch_size=2, embed_dim=192, patch_size=8, dropout=0.2, num_registers=4):

   
        super().__init__(num_classes, learning_rate, ignore_index, num_channels, num_workers, train_dataset, val_dataset, test_dataset, batch_size)

        # embed dim 192, hidden is 1536
        
        self.patch_size = patch_size
        self.num_registers = num_registers
        num_patches = (256 // patch_size) ** 2 # 256 is the image size
        self.num_patches = num_patches

        # SWIN Transformer 
        # compare the perfromance of swin vs convnextv2
        # backbone_configuration = ConvNextV2Config(num_channels=num_channels, patch_size=patch_size, image_size=256, embed_dim=embed_dim, hidden_dropout_prob= dropout, attention_probs_dropout_prob=dropout, out_features=["stage1", "stage2", "stage3", "stage4"])
        # seg_head = UperNetConfig(backbone_config=backbone_configuration, num_labels = num_classes)
        # self.swin_upernet = UperNetForSemanticSegmentation(seg_head)
        
        # backbone_configuration = Swinv2Config(num_channels=num_channels, patch_size=patch_size, image_size=256, embed_dim=embed_dim, hidden_dropout_prob= dropout, attention_probs_dropout_prob=dropout, out_features=["stage1", "stage2", "stage3", "stage4"])
        
        # self.model = Swinv2Model(backbone_configuration)
    
        backbone_configuration = Swinv2Config(
            embed_dim=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=12,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            num_channels=num_channels,
            patch_size=patch_size,
            image_size=256,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            out_features=["stage1", "stage2", "stage3", "stage4"]
        )

        # Define the segmentation head configuration
        seg_head = UperNetConfig(
            backbone_config=backbone_configuration,
            num_labels=num_classes,
            decode_head=dict(
                in_channels=[192, 384, 768, 1536],
                num_classes=num_classes,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
                    class_weight=[0.5, 1.31237, 1.38874, 1.39761, 1.5, 1.47807]
                )
            ),
            auxiliary_head=dict(
                in_channels=768,
                num_classes=num_classes,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,
                    class_weight=[0.5, 1.31237, 1.38874, 1.39761, 1.5, 1.47807]
                )
            )
        )

        # Initialize the model
        self.model = UperNetForSemanticSegmentation(seg_head)
        
        
        # print(self.swin_upernet.config)

        
       
    def forward(self, msi_img, sar_img):
         
        # print("msi_img shape", msi_img.shape)
        # outputs = self.swin_upernet(msi_img)
        
        outputs = self.model(msi_img)
        
        x = outputs
        
        print("x shape", x.shape)

        return x.logits