import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data
import glob
import torchvision.transforms as transforms
import os 
import cv2
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.data.transforms import augment, paired_random_crop
import random

def pad_to_square(data_gt,data_turb,gt_size):
    H, W, C = data_gt.shape
    # L = np.max([H,W])
    L = gt_size
    dH = L - H
    dW = L - W
    assert dH >0 or dW > 0

    data_turb = cv2.copyMakeBorder(data_turb,0,dH,0,dW,cv2.BORDER_REFLECT)
    data_gt   = cv2.copyMakeBorder(data_gt,0,dH,0,dW,cv2.BORDER_REFLECT)

    if random.random() > 2/3:
        data_gt, data_turb = data_gt[::-1,:,:], data_turb[::-1,:,:]
    if random.random() > 2/3:
        data_gt, data_turb = data_gt[:,::-1,:], data_turb[:,::-1,:]
    # print(zeros_gt.shape, zeros_turb.shape)

    return  np.ascontiguousarray(data_gt),  np.ascontiguousarray(data_turb)

class Turbulence_DATASet(data.Dataset):
    def __init__(self, opt):
        super(Turbulence_DATASet, self).__init__()
        self.opt = opt
        # print(opt) 
        # print(os.path.join(Path(opt['dataroot_gt'])) )
        self.path = sorted(glob.glob(os.path.join(Path(opt['dataroot_gt'])) + "/*turb.png"))
        # self.path = sorted(glob.glob(opt + "/*turb.png"))
        self.transform = transforms.Compose( [transforms.ToTensor(),])
    
    def __getitem__(self, index):
        index = int(index)
        path_turb = self.path[index % len(self.path)]
        path_gt = path_turb[:-8]+".png"

        data_turb = cv2.imread(path_turb)
        data_gt = cv2.imread(path_gt)
        
        # flag = 1
        try:
            data_gt.shape
        except:
        #     flag = 0
        # if self.opt['name'] != 'TrainSet' and (flag==0):
            print("data_gt to data_turb:",path_gt)
            data_gt = data_turb.copy()
        # data_turb = np.expand_dims(data_turb, axis=2)
        # data_gt = np.expand_dims(data_gt, axis=2)

        # normalization
        data_turb = data_turb.astype(np.float32) / 255.
        data_gt = data_gt.astype(np.float32) / 255.
        # print(data_turb,data_gt)

        # if self.opt['name'] == 'TrainSet':
        #     scale = self.opt['scale']
        #     gt_size = self.opt['gt_size']
        #     data_gt, data_turb = paired_random_crop(data_gt, data_turb, gt_size, scale, None)
         
        if self.opt['name'] == 'TrainSet':
            scale = self.opt['scale']
            gt_size = self.opt['gt_size']
            H, W, C = data_gt.shape
            # data_gt, data_turb = pad_to_square(data_gt, data_turb,gt_size)
            # cv2.imwrite("./"+str(random.random())+"gt.png",data_gt*255)
            data_gt, data_turb = paired_random_crop(data_gt, data_turb, gt_size, scale, None)

        data_turb = img2tensor(data_turb, bgr2rgb=False)
        data_gt = img2tensor(data_gt, bgr2rgb=False)
      
        return {'lq': data_turb, 'gt': data_gt, "lq_path": self.path[index % len(self.path)]}
    
    def __len__(self):
        return len(self.path)


class TurbSequence_DATASet(data.Dataset):
    def __init__(self, opt):
        super(TurbSequence_DATASet, self).__init__()
        self.opt = opt
        # print(opt) 
        # print(os.path.join(Path(opt['dataroot_gt'])) )
        if self.opt['name'] == 'TrainSet':
            self.path = sorted(glob.glob(os.path.join(Path(opt['dataroot_gt'])) + "/*/gt.png"))
        else:
            self.path = sorted(glob.glob(os.path.join(Path(opt['dataroot_gt'])) + "/*"))

        # self.path = sorted(glob.glob(opt + "/*/gt.png"))
        # self.path = sorted(glob.glob(opt + "/*turb.png"))
        self.transform = transforms.Compose( [transforms.ToTensor(),])
    
    def __getitem__(self, index):
        index = int(index)
        path_gt = self.path[index % len(self.path)]
        if self.opt['name'] == 'TrainSet':
            path_turbs = sorted(glob.glob(path_gt[:-6]+"turb/*"))
        else:
            path_turbs = sorted(glob.glob(path_gt+"/*"))
            # print(path_turbs)
        turb = []
        for path_turb in path_turbs[:90]:
            data_read = cv2.imread(path_turb)
            turb.append(data_read)
        data_turb = np.concatenate(turb,axis=2)
        # print(data_turb.shape)
            
        data_gt = cv2.imread(path_gt)
        # if self.opt['name'] != 'TrainSet':
        #     print("---------------------------------",data_gt.shape)
        
        # flag = 1
        try:
            data_gt.shape
        except:
        #     flag = 0
        # if self.opt['name'] != 'TrainSet' and (flag==0):
            # print("data_gt to data_turb:",path_gt)
            data_gt = data_read.copy()
            
        # data_turb = np.expand_dims(data_turb, axis=2)
        # data_gt = np.expand_dims(data_gt, axis=2)

        # normalization
        data_turb = data_turb.astype(np.float32) / 255.
        data_gt = data_gt.astype(np.float32) / 255.
        # print(data_turb,data_gt)

        # if self.opt['name'] == 'TrainSet':
        #     scale = self.opt['scale']
        #     gt_size = self.opt['gt_size']
        #     data_gt, data_turb = paired_random_crop(data_gt, data_turb, gt_size, scale, None)
         
        if self.opt['name'] == 'TrainSet':
            scale = self.opt['scale']
            gt_size = self.opt['gt_size']
            H, W, C = data_gt.shape
            # data_gt, data_turb = pad_to_square(data_gt, data_turb,gt_size)
            # cv2.imwrite("./"+str(random.random())+"gt.png",data_gt*255)
            data_gt, data_turb = paired_random_crop(data_gt, data_turb, gt_size, scale, None)

        data_turb = img2tensor(data_turb, bgr2rgb=False)
        data_gt = img2tensor(data_gt, bgr2rgb=False)
      
        return {'lq': data_turb, 'gt': data_gt, "lq_path": self.path[index % len(self.path)]}
    
    def __len__(self):
        return len(self.path)


# from torch.utils.data import DataLoader
# dataloader = DataLoader(
#     TurbSequence_DATASet(r"/mnt/data/optimal/zhangziran/dataset/n_turbulence/1_turb/train/"),
#     batch_size=1,
#     num_workers=1,
#     shuffle=True,
#     drop_last=True,)

# for i, batch in enumerate(dataloader):
#     print(batch)
#     break