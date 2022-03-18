import torch
import torch.utils.data
import torch.nn as nn


def get_one_mask(keep_index,n_pts = 2048):
    parts_num = len(keep_index)
    part_dim = int(n_pts/parts_num)
    mask = []
    for i in range(0,parts_num):
        if keep_index[i] == 1:
            tensor = torch.ones(part_dim,3)
        else:
            tensor = torch.zeros(part_dim,3)
        mask.append(tensor)
    
    mask = torch.cat(mask,dim=0)
    
    return mask

def get_mask(keep_index,n_pts = 2048):
    bs = keep_index.shape[0]
    mask= torch.zeros(bs,n_pts,3)
    for i in range(0,bs):
        line_mask = get_one_mask(keep_index[i],n_pts=n_pts)
        
        mask[i] = line_mask
    
    return mask


a = torch.tensor([[1,0,0,1],
                  [0,0,1,1]])

mask = get_mask(a,8)

print(mask.shape)
print(mask)