import glob
import os.path
import random
import numpy as np
import pandas as pd 
import torchio as tio 
import torch
import torchvision.transforms as transforms
from PIL import Image
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import nibabel as nib
from others.utils import to_pixel_samples
import math
import h5py
# ----------------------------------------------------------------------------------------------------------------------------------
# 用来读取mask
from data.select_mask import define_Mask
# Fourier变换模块
from scipy.io import *
from scipy.fftpack import *
# ----------------------------------------------------------------------------------------------------------------------------------
### 自定义 T2-PD 配对 IXI Dataset
class AlignedDataset2(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.df = pd.read_csv(self.root)
        self.df = self.df.loc[self.df['fold'] == opt.phase]  # 获取当前阶段 fold 的数据 (train / valid / test) 

       
        if self.opt.debug == 1:
            self.df = self.df.head(256)  # for debug

        # 获得mask
        self.mask = define_Mask(self.opt)


        print(f"---------------------------{opt.phase} images: {len(self.df)} ---------------------------")

    def __getitem__(self, index):
        
        s = 1
        _, slice_indice, _ , t2_volumn_path, pd_volumn_path, _ = self.df.iloc[index].tolist()
        A = tio.ScalarImage(t2_volumn_path).data[..., (slice_indice)]
        B = tio.ScalarImage(pd_volumn_path).data[..., (slice_indice)]
        mask = self.mask
        # 将mask数组扩展一个维度
        mask = torch.from_numpy(mask)
        mask_ = mask
        mask = mask.unsqueeze(0)   
        mask = self.roll(mask, 128, 1)
        mask = self.roll(mask, 128, 2)
        # print(mask.shape)9a


        ##############################################################################################################################
        # print(A.shape)
        # print(mask.shape)
        # print(A.max())
        A = (A - A.min()) / (A.max() - A.min())
        B = (B - B.min()) / (B.max() - B.min())
  

        # w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        # h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))

        # A = A[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]


        A_downsample, A_downsample_image, A_downsample_image_two = self.undersample_kspace_k(A, mask)
        B_downsample, B_downsample_image, B_downsample_image_two = self.undersample_kspace_fake_new(B, mask)
        # print(A_downsample_image_two.shape)

        _, A_downsample_fake,_, = self.undersample_kspace_fake_new(A, mask)
        img = torch.zeros_like(A_downsample_image)

       
        hr_coord, hr_rgb = to_pixel_samples(img.contiguous())


        cell = torch.ones_like(hr_coord)
        # 计算每个像素的网格大小
        cell[:, 0] *= 2 / img.shape[-2]
        cell[:, 1] *= 2 / img.shape[-1]

        mask = torch.cat((mask,mask),0)

        # print('\033[1;31m---aligned_dataset.py---\033[0m')
        # print("mask.shape = ", mask.shape)
        # mask.shape = torch.Size([2, 256, 256])

        ##############################################################################################################################
        ##############################################################################################################################

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        # if (not self.opt.no_flip) and random.random() < 0.5:
        #     idx = [i for i in range(A.size(2) - 1, -1, -1)]
        #     idx = torch.LongTensor(idx)
        #     A = A.index_select(2, idx)


        # A_k是欠采样K空间数据，A是欠采样图像，B是Ground Truth, D是辅助模态图像
        return {'A_k': A_downsample, 'A': A_downsample_image, 'A_two': A_downsample_image_two, 'B': A_downsample_fake,'C': A_downsample_fake, 'Mask': mask,'D_k': B_downsample, 'D': B_downsample_image, 'D_two': B_downsample_image_two,
                'A_paths': t2_volumn_path, 'B_paths': pd_volumn_path, 'slice_idx': int(slice_indice),'coord': hr_coord,'cell': cell,'gt': hr_rgb,'scale': s}

    def __len__(self):
        return (len(self.df))

    def name(self):
        return 'AlignedDataset2'

    def roll(self, tensor, shift, axis):
        if shift == 0:
            return tensor

        if axis < 0:
            axis += tensor.dim()

        dim_size = tensor.size(axis)
        after_start = dim_size - shift
        if shift < 0:
            after_start = -shift
            shift = dim_size - abs(shift)

        before = tensor.narrow(axis, 0, dim_size - shift)
        after = tensor.narrow(axis, after_start, shift)
        return torch.cat([after, before], axis)

    def undersample_kspace_k(self, x, mask):
        # print(x.shape)
        x_k = torch.fft.fft2(x,dim=(-2,-1))
        # print(x_k.shape)
        x_k1 = torch.stack((x_k.real, x_k.imag), dim=-1)
        x_k1_real = x_k1[...,0]
        x_k1_imag = x_k1[...,1]
        x_k_return = torch.cat((x_k1_real, x_k1_imag), dim=0)
        mask = torch.cat((mask, mask), dim=0)
        x_k_return = x_k_return * mask
        x_k1_return_real = x_k_return[0,...].unsqueeze(0)
        x_k1_return_imag = x_k_return[1,...].unsqueeze(0)
        x_k2 = torch.complex(x_k1_return_real, x_k1_return_imag)
        x_image = torch.fft.ifft2(x_k2, dim=(-2, -1))
        x_image_return = torch.cat((x_image.real,x_image.imag),0)
        x_k_image = torch.sqrt(x_image.real * x_image.real + x_image.imag * x_image.imag)
        # print(x_k_return.shape)
        # print(x_image_return.shape)
        return x_k_return,x_k_image, x_image_return

    def undersample_kspace_fake_new(self, x, mask):
        x_k = torch.fft.fft2(x,dim=(-2,-1))
        # print(x_k.shape)
        x_k1 = torch.stack((x_k.real, x_k.imag), dim=-1)
        x_k1_real = x_k1[...,0]
        x_k1_imag = x_k1[...,1]
        x_k_return = torch.cat((x_k1_real, x_k1_imag), dim=0)
        # mask = torch.cat((mask, mask), dim=0)
        # x_k_return = x_k_return * mask
        x_k1_return_real = x_k_return[0,...].unsqueeze(0)
        x_k1_return_imag = x_k_return[1,...].unsqueeze(0)
        x_k2 = torch.complex(x_k1_return_real, x_k1_return_imag)
        x_image = torch.fft.ifft2(x_k2, dim=(-2, -1))
        x_image_return = torch.cat((x_image.real,x_image.imag),0)
        x_k_image = torch.sqrt(x_image.real * x_image.real + x_image.imag * x_image.imag)
        # print(x_k_return.shape)
        # print(x_image_return.shape)
        return x_k_return,x_k_image, x_image_return
    # def undersample_kspace_fake_new(self, x, mask):
    #     x_fft = torch.fft.fft2(x,dim=(-2,-1))
    #     x_fft1 = torch.cat((x_fft.real,x_fft.imag),0)
    #     x_k_full_return = x_fft1
    #     x_image_full = torch.fft.ifft2(x_fft , dim=(-2, -1)) 
    #     # x_image_full_return = torch.cat((x_image_full.real,x_image_full.imag),0)
    #     x_image_full_return = torch.sqrt(x_image_full.real * x_image_full.real + x_image_full.imag * x_image_full.imag)
    #     # print(x_k_full_return.shape)
    #     return  x_image_full_return


# ----------------------------------------------------------------------------------------------------------------------------------
