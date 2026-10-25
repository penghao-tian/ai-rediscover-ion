"""
@Author: Penghao Tian <tph@ustc.edu.cn>
@Date: 2025/07/23 19:55
@Description: 
Perform inference with the trained model
Inference is primarily performed on a single GPU

Run model inference on a single GPU
Select a data preprocessing method from the methods defined in dataset.py
The model is defined in a class in model.py
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
from torch.utils.data import DataLoader
import scipy.stats

import model
import dataset
import train



class PrepareDataLoader():
    
    def __init__(self, 
                 return_type: str,
                 dataset_path: str,
                 process_method: int,
                 transform_method,
                 batch_size: int,
                 num_workers: int,) -> None:
        self.return_type = return_type
        self.dataset_path = Path(dataset_path)
        self.process_method = process_method
        self.transform_method = transform_method
        self.batch_size = batch_size
        self.num_workers = num_workers
        pass
    
    def prepare_dataloader(self):
        # Create the dataset
        data_set = dataset.create_dataset(self.dataset_path, 
                                          self.process_method, 
                                          self.transform_method, 
                                          self.return_type)
        
        # Create the corresponding dl
        data_dl = DataLoader(data_set, batch_size=self.batch_size,
                             num_workers=self.num_workers, shuffle=False, 
                             pin_memory=True)
        
        # Return the dataset and dataloader 
        # The dataset contains the raw data and the corresponding test set indices
        # No need to split the dataset into training and test sets
        return data_dl
    
    pass


class Evaluator():
    """
    Keep the functions decoupled
    The inference class should handle inference only; keep other functionality outside it
    """
    def __init__(self,
                 dataloader,
                 model,
                 args_config) -> None:
        self.dataloader = dataloader
        self.model = model
        self.gpu_id = 0
        self._init_train_method()
        self.args_config = args_config
        pass
    
    def _init_train_method(self):
        
        self.model = self.model.to(self.gpu_id)
        return 
    
    @torch.no_grad()
    def evaluate(self):
        # Return the inference results
        self.model.eval()
        output_list = []
        target_list = []
        for (source_dict, target) in self.dataloader:
            
            # Move all matrices in source to the GPU
            if isinstance(source_dict, dict):
                for key in source_dict.keys():
                    source_dict[key] = source_dict[key].to(self.gpu_id)
                pass
            else:
                source_dict = source_dict.to(self.gpu_id)
                pass
            
            output = self.model(source_dict)
            output = torch.squeeze(output).cpu().detach().numpy()
            output_list.append(output)
            target_list.append(target.numpy())
            pass
        print("Inference has completed.FUCK!")
        output_array = np.concatenate(output_list, axis=0)
        target_array = np.concatenate(target_list, axis=0)
        
        # Return the arrays as the corresponding NumPy objects
        return output_array, target_array
    pass
