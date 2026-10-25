"""
@Author: Penghao Tian <tph@ustc.edu.cn>
@Date: 2025/07/24 21:31
@Description: 
Train the model using data parallelism
1. Define the model architecture
2. Define the dataset
3. Define the training procedure
"""


import os
import sys
import time
import torch
import einops
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import configparser
from torch import nn
from pathlib import Path
from datetime import datetime
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


from collections import OrderedDict


import matplotlib as mpl
import matplotlib.pyplot as plt
import model
import dataset


class DefineModel():
    
    def __init__(self, 
                 model_name: str, 
                 feature_size,
                 cluster_center,
                 load_model_params_file: str) -> None:
        
        self.model = self._define_model(model_name, feature_size,
                                        cluster_center)
        
        try:
            params_file = Path(load_model_params_file)
            print(f"{load_model_params_file} has been loaded.")
            if params_file.is_file():
                ckpt = torch.load(params_file, map_location="cpu")
                self.model.to("cpu")
                self.model.load_state_dict(ckpt["model_state_dict"])
                pass
            pass
        except:
            pass
        
        return
    
    def _define_model(self, name, feature_size, cluster_center):
        
        if name == "encode_decode":
            submodel = OrderedDict()
            submodel["encode"] = model.Encoder(feature_in=feature_size)
            submodel["decode"] = model.Decoder(feature_out=feature_size)
            layer = nn.Sequential(submodel)
            pass
        elif name == "encode_cluster":
            # For cluster centers, pass the results of the previous training run; for ordinary training, this parameter should be None
            submodel = OrderedDict()
            submodel["encode"] = model.Encoder(feature_in=feature_size)
            submodel["cluster"] = model.Cluster(center=cluster_center)
            layer = nn.Sequential(submodel)
            pass
        elif name == "self_ann":
            # Use the existing model modified from ResNet
            submodel = OrderedDict()
            submodel["first_layer"] = model.FirstLinear(feature_in=feature_size)
            submodel["transform_layer"] = model.Trans1DTo2DLayerWith3Channel()
            submodel["selfann"] = model.ResNet("Bottleneck", [1, 1, 1, 1])
            submodel["last_layer"] = model.LastLinear()
            
            submodel["decode"] = model.Decoder(feature_out=feature_size)
            layer = nn.Sequential(submodel)
            
        else:
            raise OSError("undefine model structure.")
            pass
        return layer
    
    pass


class PrepareDataLoader():
    
    def __init__(self, 
                 dataset_path: str,
                 process_method: int,
                 transform_method: int,
                 return_type,
                 split_method,
                 batch_size: int,
                 num_workers: int,) -> None:
        self.dataset_path = Path(dataset_path)
        self.process_method = process_method
        self.transform_method = transform_method
        self.return_type = return_type
        self.split_method = split_method
        self.batch_size = batch_size
        self.num_workers = num_workers
        pass
    
    def prepare_dataloader(self):
        # Create the dataset
        data_set = dataset.create_dataset(self.dataset_path, 
                                          self.process_method, 
                                          self.transform_method,
                                          self.return_type)
        
        # Specify the training/validation split method
        train_ds, valid_ds = dataset.split_to_train_valid(data_set, 
                                                          split_method=self.split_method)
        
        ######
        # Note that train_ds and the distributed sampler use the same dataset, train_ds
        # Set shuffle to false; shuffling is handled by the sampler
        train_dl = DataLoader(train_ds, batch_size=self.batch_size,
                              num_workers=self.num_workers, shuffle=False, 
                              pin_memory=True, sampler=DistributedSampler(train_ds))

        valid_dl = DataLoader(valid_ds, batch_size=self.batch_size,
                              num_workers=self.num_workers, shuffle=False, 
                              pin_memory=True, sampler=DistributedSampler(valid_ds))
        
        return (train_dl, valid_dl)
    
    pass


class Trainer():
    
    def __init__(self,
                 dataloader,
                 model_name,
                 model,
                 gpu_id,
                 loss_func_name,
                 optimizer_name,
                 lr_scheduler_name,
                 lr,
                 max_epoch,
                 save_model_dir,
                 save_log_file,
                 args_config
                ):
        self.train_dl, self.valid_dl = dataloader
        self.model_name = model_name
        self.model = model
        self.gpu_id = gpu_id
        self.loss_func_name = loss_func_name
        self.optimizer_name = optimizer_name
        self.lr_scheduler_name = lr_scheduler_name
        self.lr = lr
        self.max_epoch = max_epoch
        self.save_model_dir = Path(save_model_dir)
        self.save_log_file = Path(save_log_file)
        
        self.args_config = args_config
        self._init_train_method()
        self.log = pd.DataFrame(columns=["time", "train_loss", "valid_loss", "model_name"], 
                                index=range(self.max_epoch))
        pass
    
    def _init_train_method(self):
        
        # Synchronize the BN layers
        BN_tag = False
        for mm in self.model.modules():
            if isinstance(mm, nn.BatchNorm2d):
                BN_tag = True
                break
            pass
        if BN_tag:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            pass
        
        # Use DDP
        self.model = self.model.to(self.gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        # self.model = torch.compile(self.model)
        
        # Specify the loss
        if self.loss_func_name=="L1Loss":
            self.loss_func = nn.L1Loss()
            pass
        elif self.loss_func_name=="MSELoss":
            self.loss_func = nn.MSELoss()
            pass
        elif self.loss_func_name=="CustomLoss01":
            self.loss_func = model.CustomLoss01()
            pass
        elif self.loss_func_name=="CustomLoss02":
            self.loss_func = model.CustomLoss02()
            pass
        else:
            raise OSError("undefine loss function")
        
        # Specify the optimizer
        if self.optimizer_name=="Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            pass
        elif self.optimizer_name=="Adamw":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                              weight_decay=0.00025)
            pass
        elif self.optimizer_name=="NAdam":
            self.optimizer = torch.optim.NAdam(self.model.parameters(), lr=self.lr)
            pass
        elif self.optimizer_name=="SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr,
                                             momentum=0.9, nesterov=True)
            pass
        elif self.optimizer_name=="RMSprop":
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
            pass
        else:
            raise OSError("undefine optimizer")
        
        # Specify the learning rate scheduling scheme
        if self.lr_scheduler_name=="StepLR":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                                step_size=20, gamma=0.5)
            pass
        elif self.lr_scheduler_name=="CosineAnnealingLR":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                           T_max=10, 
                                                                           eta_min=5e-6)
        else:
            self.lr_scheduler = None
            pass
        
        return 
    
    def _run_train_batch(self, source_dict, targets):
        self.optimizer.zero_grad()
        # Provide the corresponding feature dict as required by the model 
        # The feature dict is processed in model.py
        # No changes to train are needed
        output = self.model(source_dict)
        # if torch.isnan(output).any():
        #     print("Model output contains NaN!")
        
        loss = self.loss_func(output, targets)
        
        # loss = self.loss_func(output, targets)
        loss.backward()
        
        # Gradient clipping
        # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        return loss
    
    def _run_train_epoch(self, epoch):
        """
        Train over multiple epochs and average the losses collected from all GPUs
        """
        self.model.train()
        train_loss = []
        for (source_dict, targets) in self.train_dl:
            
            # Move all matrices in source to the GPU
            if isinstance(source_dict, dict):
                for key in source_dict.keys():
                    source_dict[key] = source_dict[key].to(self.gpu_id)
                pass
            else:
                source_dict = source_dict.to(self.gpu_id)
                pass
            
            targets = targets.to(self.gpu_id)
            train_loss.append(self._run_train_batch(source_dict, targets))
            pass
        train_loss = torch.stack(train_loss, axis=0)
        dist.all_reduce(train_loss, dist.ReduceOp.AVG)
        if self.gpu_id == 0:
            self.log.loc[epoch, "time"] = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
            self.log.loc[epoch, "train_loss"] = torch.mean(train_loss).to("cpu").detach().numpy()
            pass
        return
    
    @torch.no_grad()
    def _run_valid_epoch(self, epoch):
        """
        Compute the loss on the validation set
        """
        self.model.eval()
        valid_loss = []
        for (source_dict, targets) in self.valid_dl:
            
            # Move all matrices in source to the GPU
            if isinstance(source_dict, dict):
                for key in source_dict.keys():
                    source_dict[key] = source_dict[key].to(self.gpu_id)
                pass
            else:
                source_dict = source_dict.to(self.gpu_id)
                pass
            
            targets = targets.to(self.gpu_id)
            output = self.model(source_dict)
            
            loss = self.loss_func(output, targets)
            
            # loss = self.loss_func(output, targets)
            valid_loss.append(loss)
            pass
        valid_loss = torch.stack(valid_loss, axis=0)
        dist.all_reduce(valid_loss, dist.ReduceOp.AVG)
        if self.gpu_id == 0:
            self.log.loc[epoch, "valid_loss"] = torch.mean(valid_loss).to("cpu").detach().numpy()
            pass
        return
    
    def train(self):

        for epoch in range(self.max_epoch):
            self.train_dl.sampler.set_epoch(epoch)
            self.valid_dl.sampler.set_epoch(epoch)
            self._run_train_epoch(epoch)
            self._run_valid_epoch(epoch)
            if self.lr_scheduler:
                self.lr_scheduler.step()
                pass
            else:
                pass
            if self.gpu_id == 0:
                self._save_log(epoch)
                pass
            pass
        return
    
    def _save_log(self, epoch):
        
        # Record the filename for each epoch and save files during the last few epochs
        # to facilitate locating valid files later
        save_dict = {}
        save_dict["epoch"] = epoch
        save_dict["model_state_dict"] = self.model.module.state_dict()
        # save_dict["optimizer_state_dict"] = self.optimizer.state_dict()
        ckpt_name = (f"{self.model_name}_" +
                     f"{self.loss_func_name}_" +
                     f"{self.optimizer_name}_" +
                     f"{self.lr_scheduler_name}_"+
                     f"{self.args_config.getint('train_params', 'seed')}_"+
                     f"{self.lr:1.0E}_"+
                     f"{epoch:03}.ckpt")
        save_model_file = self.save_model_dir.joinpath(ckpt_name)
        
        self.log.loc[epoch, "model_name"] = save_model_file.resolve()
        self.log.to_csv(self.save_log_file, mode="w")
        
        if ((self.max_epoch - epoch) >= 0) & ((self.max_epoch - epoch) <= 20):
            torch.save(save_dict, save_model_file)
            print(f"Epoch {epoch} | Training checkpoint saved at {save_model_file}")
            pass
        
        # Plot the corresponding figures to inspect changes in the loss function
        plt.figure(figsize=(10, 5))
        plt.plot(self.log["train_loss"], label='Training Loss', marker="o")
        plt.plot(self.log["valid_loss"], label='Valid Loss', marker="o")
        plt.title(f'Epoch [{epoch+1}/{self.max_epoch}]')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        fig_file = self.save_model_dir.joinpath(f"loss_plot_{self.model_name}.png")
        plt.savefig(fig_file)
        plt.close()
    
    pass


