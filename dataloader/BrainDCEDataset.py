import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm
from scipy.io import loadmat
from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

import h5py

class DCESample(NamedTuple):
    """
    A sample of DCE data.
    """
    DCE: torch.Tensor
    MTT: torch.Tensor
    CBV: torch.Tensor
    CBF: torch.Tensor
    fname: str
    slice_num: int

class BrainDCEDataset(Dataset):
    
    def __init__(self, root_dir, file_path, IF_TRAIN= False):
        self.root_dir = root_dir
        self.data_dir = self.get_data_dir(file_path)
        self.IF_TRAIN = IF_TRAIN
        self.all_data = self.load_data()
        
    def __len__(self):
        return len(self.all_data)
    
    def get_data_dir(self, txt_path):
        data_dir = []
        with open(txt_path,'r') as f:
            tmp = f.readlines()
            for i in tmp:
                data_dir.append(i.strip('\n'))
        return data_dir

    def load_data(self):

        all_sample = []

        for idx in tqdm(range(len(self.data_dir))):
            case_name = self.data_dir[idx]

            with h5py.File(self.root_dir+'/'+case_name+'.h5') as hf:

                DCE = torch.Tensor(hf['DCE'][:])
                MTT = torch.Tensor(hf['MTT'][:])
                CBF = torch.Tensor(hf['rCBF'][:])
                CBV = torch.Tensor(hf['rCBV'][:])

                for slice_idx in range(DCE.shape[0]):
                    DCE_slice = DCE[slice_idx]
                    DCE_slice[DCE_slice<0]=0

                    DCE_slice = -1+2*(DCE_slice-DCE_slice.min())/(DCE_slice.max()-DCE_slice.min())

                    MTT_slice = MTT[slice_idx]
                    CBV_slice = CBV[slice_idx]
                    CBF_slice = CBF[slice_idx]

                    MTT_slice[MTT_slice>=30] = 30
                    MTT_slice[MTT_slice<=0] = 0
                    MTT_slice = -1+2*(MTT_slice-0)/(30-0)

                    CBF_slice = -1+2*(CBF_slice-CBF_slice.min())/(CBF_slice.max()-CBF_slice.min())
                    
                    CBV_slice = -1+2*(CBV_slice-CBV_slice.min())/(CBV_slice.max()-CBV_slice.min())

                    sample = DCESample(DCE = DCE_slice,
                                        MTT = MTT_slice,
                                        CBV = CBV_slice,
                                        CBF =  CBF_slice,
                                        fname = case_name,
                                        slice_num = slice_idx
                                        )
                    all_sample.append(sample)

        return all_sample

    def __getitem__(self,idx):
        
        sample = self.all_data[idx]
        
        return sample

class BrainDCEJointDataset(Dataset):
    
    def __init__(self, root_dir, file_path, IF_TRAIN= False):
        self.root_dir = root_dir
        self.data_dir = self.get_data_dir(file_path)
        self.IF_TRAIN = IF_TRAIN
        self.all_data = self.load_data()
        
    def __len__(self):
        return len(self.all_data)
    
    def get_data_dir(self, txt_path):
        data_dir = []
        with open(txt_path[0],'r') as f:
            tmp = f.readlines()
            for i in tmp:
                data_dir.append(self.root_dir[0]+i.strip('\n'))

        with open(txt_path[1],'r') as f:
            tmp = f.readlines()
            for i in tmp:
                data_dir.append(self.root_dir[1]+i.strip('\n'))

        return data_dir

    def load_data(self):

        all_sample = []

        for idx in tqdm(range(len(self.data_dir))):
            case_name = self.data_dir[idx]

            with h5py.File(case_name+'.h5') as hf:

                DCE = torch.Tensor(hf['DCE'][:])
                MTT = torch.Tensor(hf['MTT'][:])
                CBF = torch.Tensor(hf['rCBF'][:])
                CBV = torch.Tensor(hf['rCBV'][:])
                
                b,t,w,h = DCE.shape

                DCE = torch.nn.functional.interpolate(DCE[:,None,...], size=(150,w,h))[:,0,...]
                MTT = torch.nn.functional.interpolate(MTT[:,None,...], size=(w,h))[:,0,...]
                CBF = torch.nn.functional.interpolate(CBF[:,None,...], size=(w,h))[:,0,...]
                CBV = torch.nn.functional.interpolate(CBV[:,None,...], size=(w,h))[:,0,...]

                for slice_idx in range(DCE.shape[0]):
                    
                    DCE_slice = DCE[slice_idx]

                    DCE_slice[DCE_slice<0]=0
                    DCE_slice = -1+2*(DCE_slice-DCE_slice.min())/(DCE_slice.max()-DCE_slice.min())

                    MTT_slice = MTT[slice_idx]
                    CBV_slice = CBV[slice_idx]
                    CBF_slice = CBF[slice_idx]
                    
                    MTT_slice[MTT_slice>=30] = 30
                    MTT_slice[MTT_slice<=0] = 0
                    MTT_slice = -1+2*(MTT_slice-0)/(30-0)

                    CBF_slice = -1+2*(CBF_slice-CBF_slice.min())/(CBF_slice.max()-CBF_slice.min())
                    
                    CBV_slice = -1+2*(CBV_slice-CBV_slice.min())/(CBV_slice.max()-CBV_slice.min())

                    sample = DCESample(DCE = DCE_slice,
                                        MTT = MTT_slice,
                                        CBV = CBV_slice,
                                        CBF =  CBF_slice,
                                        fname = case_name,
                                        slice_num = slice_idx
                                        )
                    all_sample.append(sample)

        return all_sample

    def __getitem__(self,idx):
        
        sample = self.all_data[idx]
        
        return sample