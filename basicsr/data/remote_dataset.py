import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop,four_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.flow_util import dequantize_flow


import glob
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import torch.nn.functional as F
import cv2 
from random import random, gauss
def unsqueeze_twice(x):
    return x.unsqueeze(0).unsqueeze(0)

def warp(img,jit):
    jit = torch.from_numpy(-jit)
    img = torch.from_numpy(img).float()
    h, w = img.shape
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(img)
    grid.requires_grad = False

    grid_flow = grid + jit
    grid_flow = grid_flow.unsqueeze(0)
    grid_flow = grid_flow[:, :h, :w, :]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0  
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim = 3)

    img_tensor = unsqueeze_twice(img)
    # print(img_tensor,grid_flow)
    img_subdivision = F.grid_sample(img_tensor, grid_flow, 
        mode = 'bilinear', padding_mode = "reflection", align_corners = True) # nearest
    img = np.array(img_subdivision).astype(int)[0,0,:,:]
    return img

class REMOTEDATASet(data.Dataset):
    def __init__(self, opt):
        super(REMOTEDATASet, self).__init__()
        self.opt = opt
        print(opt) 
        self.path = sorted(glob.glob(os.path.join(Path(opt['dataroot_gt'])) + "/*.*")) #+ sorted(glob.glob( "/workspace/zhangzr/project/origin_data/TDI_dataset_td_2022_12_12/tdi_dataset/test/*.*"))[:2000]
        self.transform = transforms.Compose( [transforms.ToTensor(),])
    
    def __getitem__(self, index):
        index = int(index)
        data_dict = np.load(self.path[index % len(self.path)], allow_pickle=True )
        data_dict = data_dict.item()

        img = data_dict["img_TDI"]
        gt = data_dict["img_gt"]
        flow = data_dict["jit_information_noise"]
        flow_gt = data_dict["jit_information"]
        
        # img = warp(img,flow)

        img = np.expand_dims(img, axis=2)
        gt = np.expand_dims(gt, axis=2)
        
        # print(np.max(img))

        # normalization
        img = img.astype(np.float32) / (255. * 255.)
        gt = gt.astype(np.float32) / (255. * 255.)
     
        if self.opt['name'] == 'TrainSet':
            scale = self.opt['scale']
            gt_size = self.opt['gt_size']
            # gt, img = paired_random_crop(gt, img, gt_size, scale, None) 
            gt, img, flow, flow_gt = four_random_crop(gt, img, flow, flow_gt, gt_size, scale, None) 
            # if np.random.rand(1) <= (1/3.0):
                # flow = np.zeros_like(flow)
            # print(22222222222222222222222222)
        

 
        # print("dataset",flow_gt,flow)
        img  = img2tensor(img)
        gt   = img2tensor(gt)
        flow = img2tensor(flow)
        flow_gt = img2tensor(flow_gt)

        return {'lq': img, 'gt': gt, 'flow': flow, "flow_gt": flow_gt,"lq_path": self.path[index % len(self.path)]}

    def __len__(self):
        return len(self.path)

class REMOTEDATASetWarp(data.Dataset):
    def __init__(self, opt):
        super(REMOTEDATASetWarp, self).__init__()
        self.opt = opt
        print(opt) 
        self.path = sorted(glob.glob(os.path.join(Path(opt['dataroot_gt'])) + "/*.*")) #+ sorted(glob.glob( "/workspace/zhangzr/project/origin_data/TDI_dataset_td_2022_12_12/tdi_dataset/test/*.*"))[:2000]
        self.transform = transforms.Compose( [transforms.ToTensor(),])
    
    def __getitem__(self, index):
        index = int(index)
        data_dict = np.load(self.path[index % len(self.path)], allow_pickle=True )
        data_dict = data_dict.item()

        img = data_dict["img_TDI"]
        gt = data_dict["img_gt"]
        flow = data_dict["jit_information_noise"]
        flow_gt = data_dict["jit_information"]
        
        # normalization
        img = img.astype(np.float32) / (255. )
        gt = gt.astype(np.float32) / (255. )
        img = warp(img.astype(np.float32),flow)

        img = np.expand_dims(img, axis=2)
        gt = np.expand_dims(gt, axis=2)
        
        # print(np.max(img))
        img = img / ( 255.)
        gt = gt / (255.)
     
        if self.opt['name'] == 'TrainSet':
            scale = self.opt['scale']
            gt_size = self.opt['gt_size']
            # gt, img = paired_random_crop(gt, img, gt_size, scale, None) 
            gt, img, flow, flow_gt = four_random_crop(gt, img, flow, flow_gt, gt_size, scale, None) 
        # if np.random.rand(1) <= (1/3.0):
        #     flow = np.zeros_like(flow)
        #     # print(22222222222222222222222222)
        

 
        # print("dataset",flow_gt,flow)
        img  = img2tensor(img)
        gt   = img2tensor(gt)
        flow = img2tensor(flow)
        flow_gt = img2tensor(flow_gt)

        return {'lq': img, 'gt': gt, 'flow': flow, "flow_gt": flow_gt,"lq_path": self.path[index % len(self.path)]}

    def __len__(self):
        return len(self.path)



# dataloader = DataLoader(
#     REMOTEDATASet(r"/workspace/zhangzr/project/origin_data/TDI_dataset_td_2022_12_12/tdi_dataset/train/"),
#     batch_size=1,
#     num_workers=1,
#     shuffle=True,
#     drop_last=True,)

# for i, batch in enumerate(dataloader):
#     print(batch)
#     break

# class REDSDataset(data.Dataset):
    
#     def __init__(self, opt):
#         super(REDSDataset, self).__init__()
#         self.opt = opt
#         self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(
#             opt['dataroot_lq'])
#         self.flow_root = Path(
#             opt['dataroot_flow']) if opt['dataroot_flow'] is not None else None
#         assert opt['num_frame'] % 2 == 1, (
#             f'num_frame should be odd number, but got {opt["num_frame"]}')
#         self.num_frame = opt['num_frame']
#         self.num_half_frames = opt['num_frame'] // 2

#         self.keys = []
#         with open(opt['meta_info_file'], 'r') as fin:
#             for line in fin:
#                 folder, frame_num, _ = line.split(' ')
#                 self.keys.extend(
#                     [f'{folder}/{i:08d}' for i in range(int(frame_num))])

#         # remove the video clips used in validation
#         if opt['val_partition'] == 'REDS4':
#             val_partition = ['000', '011', '015', '020']
#         elif opt['val_partition'] == 'official':
#             val_partition = [f'{v:03d}' for v in range(240, 270)]
#         else:
#             raise ValueError(
#                 f'Wrong validation partition {opt["val_partition"]}.'
#                 f"Supported ones are ['official', 'REDS4'].")
#         self.keys = [
#             v for v in self.keys if v.split('/')[0] not in val_partition
#         ]

#         # file client (io backend)
#         self.file_client = None
#         self.io_backend_opt = opt['io_backend']
#         self.is_lmdb = False
#         if self.io_backend_opt['type'] == 'lmdb':
#             self.is_lmdb = True
#             if self.flow_root is not None:
#                 self.io_backend_opt['db_paths'] = [
#                     self.lq_root, self.gt_root, self.flow_root
#                 ]
#                 self.io_backend_opt['client_keys'] = ['lq', 'gt', 'flow']
#             else:
#                 self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
#                 self.io_backend_opt['client_keys'] = ['lq', 'gt']

#         # temporal augmentation configs
#         self.interval_list = opt['interval_list']
#         self.random_reverse = opt['random_reverse']
#         interval_str = ','.join(str(x) for x in opt['interval_list'])
#         logger = get_root_logger()
#         logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
#                     f'random reverse is {self.random_reverse}.')

#     def __getitem__(self, index):
#         if self.file_client is None:
#             self.file_client = FileClient(
#                 self.io_backend_opt.pop('type'), **self.io_backend_opt)

#         scale = self.opt['scale']
#         gt_size = self.opt['gt_size']
#         key = self.keys[index]
#         clip_name, frame_name = key.split('/')  # key example: 000/00000000
#         center_frame_idx = int(frame_name)

#         # determine the neighboring frames
#         interval = random.choice(self.interval_list)

#         # ensure not exceeding the borders
#         start_frame_idx = center_frame_idx - self.num_half_frames * interval
#         end_frame_idx = center_frame_idx + self.num_half_frames * interval
#         # each clip has 100 frames starting from 0 to 99
#         while (start_frame_idx < 0) or (end_frame_idx > 99):
#             center_frame_idx = random.randint(0, 99)
#             start_frame_idx = (
#                 center_frame_idx - self.num_half_frames * interval)
#             end_frame_idx = center_frame_idx + self.num_half_frames * interval
#         frame_name = f'{center_frame_idx:08d}'
#         neighbor_list = list(
#             range(center_frame_idx - self.num_half_frames * interval,
#                   center_frame_idx + self.num_half_frames * interval + 1,
#                   interval))
#         # random reverse
#         if self.random_reverse and random.random() < 0.5:
#             neighbor_list.reverse()

#         assert len(neighbor_list) == self.num_frame, (
#             f'Wrong length of neighbor list: {len(neighbor_list)}')

#         # get the GT frame (as the center frame)
#         if self.is_lmdb:
#             img_gt_path = f'{clip_name}/{frame_name}'
#         else:
#             img_gt_path = self.gt_root / clip_name / f'{frame_name}.png'
#         img_bytes = self.file_client.get(img_gt_path, 'gt')
#         img_gt = imfrombytes(img_bytes, float32=True)

#         # get the neighboring LQ frames
#         img_lqs = []
#         for neighbor in neighbor_list:
#             if self.is_lmdb:
#                 img_lq_path = f'{clip_name}/{neighbor:08d}'
#             else:
#                 img_lq_path = self.lq_root / clip_name / f'{neighbor:08d}.png'
#             img_bytes = self.file_client.get(img_lq_path, 'lq')
#             img_lq = imfrombytes(img_bytes, float32=True)
#             img_lqs.append(img_lq)

#         # get flows
#         if self.flow_root is not None:
#             img_flows = []
#             # read previous flows
#             for i in range(self.num_half_frames, 0, -1):
#                 if self.is_lmdb:
#                     flow_path = f'{clip_name}/{frame_name}_p{i}'
#                 else:
#                     flow_path = (
#                         self.flow_root / clip_name / f'{frame_name}_p{i}.png')
#                 img_bytes = self.file_client.get(flow_path, 'flow')
#                 cat_flow = imfrombytes(
#                     img_bytes, flag='grayscale',
#                     float32=False)  # uint8, [0, 255]
#                 dx, dy = np.split(cat_flow, 2, axis=0)
#                 flow = dequantize_flow(
#                     dx, dy, max_val=20,
#                     denorm=False)  # we use max_val 20 here.
#                 img_flows.append(flow)
#             # read next flows
#             for i in range(1, self.num_half_frames + 1):
#                 if self.is_lmdb:
#                     flow_path = f'{clip_name}/{frame_name}_n{i}'
#                 else:
#                     flow_path = (
#                         self.flow_root / clip_name / f'{frame_name}_n{i}.png')
#                 img_bytes = self.file_client.get(flow_path, 'flow')
#                 cat_flow = imfrombytes(
#                     img_bytes, flag='grayscale',
#                     float32=False)  # uint8, [0, 255]
#                 dx, dy = np.split(cat_flow, 2, axis=0)
#                 flow = dequantize_flow(
#                     dx, dy, max_val=20,
#                     denorm=False)  # we use max_val 20 here.
#                 img_flows.append(flow)

#             # for random crop, here, img_flows and img_lqs have the same
#             # spatial size
#             img_lqs.extend(img_flows)

#         # randomly crop
#         img_gt, img_lqs = paired_random_crop(img_gt, img_lqs, gt_size, scale,
#                                              img_gt_path)
#         if self.flow_root is not None:
#             img_lqs, img_flows = img_lqs[:self.num_frame], img_lqs[self.
#                                                                    num_frame:]

#         # augmentation - flip, rotate
#         img_lqs.append(img_gt)
#         if self.flow_root is not None:
#             img_results, img_flows = augment(img_lqs, self.opt['use_flip'],
#                                              self.opt['use_rot'], img_flows)
#         else:
#             img_results = augment(img_lqs, self.opt['use_flip'],
#                                   self.opt['use_rot'])

#         img_results = img2tensor(img_results)
#         img_lqs = torch.stack(img_results[0:-1], dim=0)
#         img_gt = img_results[-1]

#         if self.flow_root is not None:
#             img_flows = img2tensor(img_flows)
#             # add the zero center flow
#             img_flows.insert(self.num_half_frames,
#                              torch.zeros_like(img_flows[0]))
#             img_flows = torch.stack(img_flows, dim=0)

#         # img_lqs: (t, c, h, w)
#         # img_flows: (t, 2, h, w)
#         # img_gt: (c, h, w)
#         # key: str
#         if self.flow_root is not None:
#             return {'lq': img_lqs, 'flow': img_flows, 'gt': img_gt, 'key': key}
#         else:
#             return {'lq': img_lqs, 'gt': img_gt, 'key': key}

#     def __len__(self):
#         return len(self.keys)
