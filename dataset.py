"""
@Author: Penghao Tian <tph@ustc.edu.cn>
@Date: 2025/07/26 18:59
@Description:
Build the dataset
Implement different data preprocessing workflows

The create_dataset marked MAIN is the main workflow
MyDataset2 is used to simply add dwidz
"""


import torch
import pickle
import einops
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from torch.utils.data import Dataset, random_split
from sklearn.preprocessing import StandardScaler

import torchvision

    
## MAIN ##
def create_dataset(file_path, process_method, transform_method,
                   return_type):
    """
    Create the dataset object
    file_path: Path to the data file to process
    process_method: Preprocessing method selection
    transform_method: Data transformation/data augmentation method
    return_type: Data subset to return: all/training/test data
    
    """
    file_path = Path(file_path)
    assert file_path.is_file()
    
    # Preprocess the raw data
    raw_df = preprocess_raw_data(file_path)
    
    # Create the dataset structure
    my_dataset = MyDataset(raw_df, process_method, 
                           transform_method, return_type)
    return my_dataset


def preprocess_raw_data(file_path):
    """
    Read the preprocessed data
    """
    with xr.open_dataset(file_path) as ds:
        pass
    # nds = ds.isel(m_alt=slice(1,None))
    # nds.to_netcdf("./data/FUCKAAA.nc")
    # print(ds.coords["m_alt"].values)
    
    # ds = ds.isel(n_time=slice(1000000))
    
    return ds


class MyDataset(Dataset):
    """
    The inputs are s4 indices at different altitudes, and the outputs are the corresponding data
    """
    def __init__(self, raw_df, process_method, transform_method,
                 return_type):
        self.raw_ds = raw_df
        self.process_method = process_method
        self.transform_method = transform_method
        self.return_type = return_type
        self._init_dataset()
        pass
    
    def _init_dataset(self):
        
        # Preprocessing method
        if self.process_method == 1:
            self._process_method_1()
            pass
        elif self.process_method == 2:
            self._process_method_2()
            pass
        elif self.process_method == 3:
            self._process_method_3()
            pass
        elif self.process_method == 4:
            self._process_method_4()
            pass
        else:
            raise OSError("Opps")
            pass
        
        self.transform = None
        return
    
    def _process_method_1(self,):
        """
        Return all s4 index data 
        Scaled inputs
        Raw outputs
        """
        raw_s4 = self.raw_ds["s4"].values.astype(np.float32)
        
        # Split into training and test sets
        all_index = np.arange(raw_s4.shape[0])
        test_index = np.random.choice(all_index, 
                                      size=int(0.05*len(all_index)),
                                      replace=False)
        # Use the remaining data as the training set
        train_index = np.delete(all_index, test_index)
        
        # Fit the training data
        scaler = StandardScaler()
        scaler.fit(raw_s4[train_index])
        print(scaler.mean_.shape, scaler.scale_.shape)
        
        # Generate the standardized data
        scaled_s4 = scaler.transform(raw_s4).astype(np.float32)
        
        scaled_s4 = torch.from_numpy(scaled_s4)
        raw_s4 = torch.from_numpy(raw_s4)
        
        self.test_index = test_index
        self.train_index = train_index
        
        if self.return_type == "train":
            self.raw_s4 = raw_s4[self.train_index]
            self.scaled_s4 = scaled_s4[self.train_index]
            pass
        elif self.return_type == "all":
            self.raw_s4 = raw_s4
            self.scaled_s4 = scaled_s4
            pass
        else:
            raise OSError("~")
            pass
        pass
    
    def _process_method_2(self,):
        """
        Return all s4 index data 
        Raw inputs
        Raw outputs
        """
        raw_s4 = self.raw_ds["s4"].values.astype(np.float32)
        
        # Split into training and test sets
        all_index = np.arange(raw_s4.shape[0])
        test_index = np.random.choice(all_index, 
                                      size=int(0.05*len(all_index)),
                                      replace=False)
        # Use the remaining data as the training set
        train_index = np.delete(all_index, test_index)
        
        raw_s4 = torch.from_numpy(raw_s4)
        
        self.test_index = test_index
        self.train_index = train_index
        
        if self.return_type == "train":
            self.raw_s4 = raw_s4[self.train_index]
            self.scaled_s4 = self.raw_s4
            pass
        elif self.return_type == "all":
            self.raw_s4 = raw_s4
            self.scaled_s4 = self.raw_s4
            pass
        else:
            raise OSError("~")
            pass
        pass
    
    def __len__(self):
        return self.raw_s4.shape[0]
    
    def __getitem__(self, index):
        feature = self.scaled_s4[index]
        label = self.raw_s4[index]
        
        return (feature, label)
    pass



class MySubDataset(MyDataset):
    """
    Inherit from MyDataset and modify selected features only in
    the __getitem__ method to achieve the desired behavior
    
    For example, return only selected features
    This can also be implemented through process_method
    """
    pass


def split_to_train_valid(data_set, split_method=1):
    """
    Split into training and validation sets
    """
    if split_method==1:
        # Randomly select 1/10 of the data as the validation set
        n_train = int(len(data_set)*0.99)
        n_valid = len(data_set) - n_train
        dataset_train, dataset_valid = random_split(data_set, [n_train, n_valid])
        pass
    
    else:
        pass
    
    return dataset_train, dataset_valid


class Auxiliary():
    """
    Functions for data preprocessing
    """
    
    pass

    

if __name__=="__main__":
    file_path="./data/cosmic_1_2_s4_2007_2025.nc"
    # feature_size 129
    
    np.random.seed(3406)
    obj = create_dataset(file_path=file_path,
                         process_method=1,
                         transform_method=0,
                         return_type="train")
    # print(obj.test_index[34:44])
    breakpoint()
    
    