from lib2to3.pgen2.pgen import generate_grammar
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import numpy as np
import h5py

from visdom import Visdom
from pprint import pprint, pformat

from util.plot import *
# from dataset.chair_dataset import PartNetdataset
from networks.network import Generator_fc,Discriminator
from util.GradientPenalty import GradientPenalty

def get_one_mask(keep_index,n_pts = 2048):
    parts_num = len(keep_index)
    part_dim = n_pts/parts_num
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
        line_mask = get_one_mask(keep_index[i],n_pts=2048)
        
        mask[i] = line_mask
    return mask

def write_points_h5(pc,save_path):
    with h5py.File(save_path,'w') as f:
        f.create_dataset("fake_pc",data=pc)

class Runner(nn.Module):
    def __init__(self, opt,parts_num=4):
        super(Runner, self).__init__()
        self.opt = opt 
        self.generator = []
        for i in range(0,parts_num):
            self.generator[i] = torch.nn.DataParallel(Generator_fc().to(opt.device), device_ids=opt.gpu_ids)
        self.discriminator = torch.nn.DataParallel(Discriminator().to(opt.device), device_ids=opt.gpu_ids)
        self.display_id = opt.display_id
        self.vis = Visdom(env='%s' % self.display_id+'_'+str(opt.gpu_ids[0]))

        self.loss_gp = GradientPenalty(lambdaGP=10, gamma=1, device=opt.device)

        self.optimizerG = []
        for i in range(0,parts_num):
            self.optimizerG[i] = torch.optim.Adam(filter(lambda p: p.requires_grad, self.generator.parameters()), lr=opt.lr, betas=(0.5, 0.999))

        self.optimizerD = torch.optim.Adam(filter(lambda p: p.requires_grad, self.discriminator.parameters()), lr=opt.lr, betas=(0.5, 0.999))

        self.all_gen = []
        self.all_real = []

    def load_pretrained(self):
        ckpt = torch.load(self.opt.ckpt_path, map_location=lambda storage, location: storage)
        print("load epoch: %d" % (ckpt['epoch']))
        for i in range(0,len(self.generator)):
            self.generator[i].load_state_dict(ckpt['state_dict'][i])
        

    def train_func(self,data,args):
        start = time.time()

        raw_pc = data['raw'].to(self.opt.device)
        real_pc = data['real'].to(self.opt.device)
        parts_keep_index = data['parts_keep_index'] 

        B, N, _ = raw_pc.shape
        if args[1] % 5 - 1 == 0:
            parts_pc = []
            for i in range(0,len(self.generator)):
                z = torch.normal(mean=0.0, std=0.2, size=(B, 32)).float().to(self.opt.device)
                parts_pc.append(self.generator[i](z))
            
            gen_pc = torch.cat(parts_pc,dim=1)
            parts_mask = get_mask(parts_keep_index).to(self.opt.device)
            select_pc = (gen_pc*parts_mask).transpose(2,1)
            self.fake_pc = select_pc+raw_pc
            
            fake_out = self.discriminator(self.fake_pc)
            self.loss_G = -torch.mean(fake_out)
            for i in range(0,len(self.optimizerG)):
                self.optimizerG.zero_grad()
                self.loss_G.backward()
                self.optimizerG.step()


        real_out = self.discriminator(real_pc)
        fake_out = self.discriminator(self.fake_pc.detach())
        self.loss_D_real = -torch.mean(real_out)
        self.loss_D_fake = torch.mean(fake_out)
        self.loss_D_gp = self.loss_gp(self.discriminator, real_pc, self.fake_pc.detach())
        self.loss_D = self.loss_D_real + self.loss_D_fake + self.loss_D_gp

        self.optimizerD.zero_grad()
        self.loss_D.backward(retain_graph=True)
        self.optimizerD.step()

        end = time.time()

        if args[1] % 10 == 0:
            print('[%d/%d][%d/%d]' % (args[0], self.opt.n_epochs, args[1], args[2]), end=' ')
            print('Loss: G: %.6f D: %.6f Time: %.6f' % (self.loss_G, self.loss_D, end - start))
            with open(os.path.join(self.opt.save_path, 'runlog.txt'), 'a') as f:
                f.write('[%d/%d][%d/%d]' % (args[0], self.opt.n_epochs, args[1], args[2]))
                f.write('Loss: G: %.6f D: %.6f Time: %.6f\n' % (self.loss_G, self.loss_D, end - start))


    def after_one_epoch(self, args):
        self.epoch = args[0]

        losses = {'loss_G': self.loss_G.item(), 'loss_D': self.loss_D.item()}
        # plot_loss_curves(self, losses, vis=self.vis, win='loss_curves')

        if self.epoch % 50 == 0:
            save_fn = 'epoch_%d.pth' % (self.epoch)
            save_dir = os.path.join(self.opt.save_path, 'checkpoints')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            Generator_state_dict = []
            for i in range(0,len(self.generator)):
                Generator_state_dict.append(self.generator[i].state_dict())
            torch.save({'epoch': self.epoch, 'state_dict': Generator_state_dict}, os.path.join(save_dir, save_fn))

    def val_func(self,data):
        with torch.no_grad():
            raw_pc = data['raw'].to(self.opt.device)
            real_pc = data['real'].to(self.opt.device)
            parts_keep_index = data['parts_keep_index'] 

            B, N, _ = raw_pc.shape
        
            parts_pc = []
            for i in range(0,len(self.generator)):
                z = torch.normal(mean=0.0, std=0.2, size=(B, 32)).float().to(self.opt.device)
                parts_pc.append(self.generator[i](z))
            
            gen_pc = torch.cat(parts_pc,dim=1)
            parts_mask = get_mask(parts_keep_index).to(self.opt.device)
            select_pc = (gen_pc*parts_mask).transpose(2,1)
            fake_pc = select_pc+raw_pc

            self.all_gen.append(fake_pc)
            self.all_real.append(real_pc)

    def val_model(self,args):
        gen_pcs = torch.cat(self.all_sample, dim=0)
        rreal_pcs = torch.cat(self.all_ref, dim=0)

        self.all_sample = []
        self.all_ref = []
            
    def test_func(self,data):
         with torch.no_grad():
            raw_pc = data['raw'].to(self.opt.device)
            parts_keep_index = data['parts_keep_index']
            anno_id = data['anno_id']

            B, N, _ = raw_pc.shape
        
            parts_pc = []
            for i in range(0,len(self.generator)):
                z = torch.normal(mean=0.0, std=0.2, size=(B, 32)).float().to(self.opt.device)
                parts_pc.append(self.generator[i](z))
            
            gen_pc = torch.cat(parts_pc,dim=1)
            parts_mask = get_mask(parts_keep_index).to(self.opt.device)
            select_pc = (gen_pc*parts_mask).transpose(2,1)
            fake_pc = select_pc+raw_pc
            fake_pc = fake_pc.transpose(2,1)

            save_dir = self.opt.save_pc_path
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            for i in range(0,B):
                save_path = os.path.join(save_dir,anno_id)
                write_points_h5(fake_pc[i],save_path)

        