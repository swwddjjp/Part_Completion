from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import json
import random
import trimesh

import sys
sys.path.append('/root/code/multimodal shape completion via semantic decoupling/util')
from pc_utils import sample_point_cloud_by_n
import h5py
import visdom

part_name = ['chair_back', 'chair_seat', 'chair_base', 'chair_arm']

def read_h5file(path):
    part_pc = {}
    parts = []
    with h5py.File(path,'r') as f:
        for k in f.keys():
            name = f[k].name[1:]
            part_pc[name] = f[k].value
            if name in part_name:
                parts.append(name)

    return part_pc,parts

def get_exist_index(keep_part):
    exist_mask = []
    for i in range(0,len(part_name)):
        if part_name[i] in keep_part:
            exist_mask.append(1)
        else:
            exist_mask.append(0)
    return exist_mask

def random_keep_parts(phase,parts_pc,exist_parts):
    if phase == "train":
        random.shuffle(exist_parts)
        n_part_keep = random.randint(1, max(1, len(exist_parts) - 1))
    else:
        random.Random(1234).shuffle(exist_parts)
        n_part_keep = random.Random(1234).randint(1, max(1, len(exist_parts) - 1))
    parts_name_keep = exist_parts[:n_part_keep]

    keep_pc = []
    for name in parts_name_keep:
        keep_pc.extend(parts_pc[name])

    return keep_pc,parts_name_keep,n_part_keep

class PartNetdataset(Dataset):
    def __init__(self,phase,data_root,category,n_pts = 2048):
        super(PartNetdataset,self).__init__()

        if phase == "validation":
            phase = "val"

        self.phase = phase
        self.aug = phase == "train"

        path = os.path.join(data_root,category,phase)
        self.data_root = path
        files = os.listdir(path)
        self.files = files
        self.cate = category

        self.n_pts = n_pts
        self.raw_n_pts = self.n_pts // 2

        self.rng = random.Random(1234)
    
    def random_rm_parts(self,parts_pc,exist_parts):
        if self.phase == "train":
            random.shuffle(exist_parts)
            n_part_keep = random.randint(1, max(1, len(exist_parts) - 1))
        else:
            self.rng.shuffle(exist_parts)
            n_part_keep = self.rng.randint(1, max(1, len(exist_parts) - 1))
        parts_name_keep = exist_parts[:n_part_keep]

        keep_pc = []
        for name in parts_name_keep:
            keep_pc.extend(parts_pc[name])

        return keep_pc,parts_name_keep,n_part_keep

    def __getitem__(self, index):
        h5_path = os.path.join(self.data_root,self.files[index])
        all_pc,exist_parts = read_h5file(h5_path)

        raw_pc,parts_name_keep,n_part_keep = self.random_rm_parts(all_pc,exist_parts)
        raw_pc = np.array(raw_pc)
        raw_pc = sample_point_cloud_by_n(raw_pc, self.raw_n_pts)
        parts_keep_index = get_exist_index(parts_name_keep)
        raw_pc = torch.tensor(raw_pc, dtype=torch.float32).transpose(1, 0)

        real_pc = all_pc[self.cate]
        real_pc = np.array(real_pc)
        real_pc = sample_point_cloud_by_n(real_pc, self.n_pts)
        real_pc = torch.tensor(real_pc, dtype=torch.float32).transpose(1, 0)
        
        return {"raw": raw_pc, "real": real_pc,'parts_keep_index': parts_keep_index,'exist_parts':exist_parts,
                        'parts_name_keep':parts_name_keep,'n_part_keep':n_part_keep,'anno_id':self.files[index]}

    def __len__(self):
        return len(self.files)


if __name__ == "__main__":
    # path = '/root/data/PartNet_v4.0/Chair/train/39885.h5'
    # all_pc,exist_parts = read_h5file(path)

    # for name in all_pc.keys():
    #     print(name)
    #     vis=visdom.Visdom(env=name)
    #     vis.scatter(all_pc[name],opts={'markersize':3,'title':'try'},win="p")
    
    # print(exist_parts)

    # raw_pc,parts_name_keep,n_part_keep = random_keep_parts("train",all_pc,exist_parts)

    # vis=visdom.Visdom(env='raw_pc')
    # vis.scatter(raw_pc,opts={'markersize':3,'title':'try'},win="p")
    # print(parts_name_keep,n_part_keep)

    # print(get_exist_index(parts_name_keep))
    dataset = PartNetdataset("train","/root/data/PartNet_v4.0","Chair",2048)
    data = dataset[0]
    for key in data.keys():
        print(key)
        print(data[key])
        
        if key == "raw" or key == 'real':
            vis=visdom.Visdom(env=key)
            vis.scatter(data[key].transpose(1, 0),opts={'markersize':3,'title':'test'},win="p")
