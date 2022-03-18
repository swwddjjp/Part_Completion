import os
import argparse
import torch
import torch.utils.data
import torch.nn as nn
from datetime import datetime, timedelta

from dataset.chair_dataset import PartNetdataset
from runner import Runner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=list, default=[0], help='gpu_ids seperated by comma')
    parser.add_argument('--phase', type=str, default='train', help='train | test')
    parser.add_argument('--method', type=str, default='original', help='original | axform')
    parser.add_argument('--category', type = str,default="Chair", help='category names | None')
    parser.add_argument('--visual', type=bool, default=True, help='visualization during training')
    parser.add_argument('--ckpt_path', type=str, default="",help='path to checkpoints')
    parser.add_argument('--n_pts', type=int,default=2048, help='number of points used to train')

    parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
    parser.add_argument('--n_epochs', type=int, default=600, help='the epoch number of training the model')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')
    parser.add_argument('--dataroot', default='/root/data/PartNet_v4.0', help='path to point clouds')
    parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
    parser.add_argument('--display_id', type=str, default='gpu', help='window id of the web display')

    opt = parser.parse_args()
    torch.cuda.set_device('cuda:'+str(opt.gpu_ids[0]))
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = PartNetdataset(opt.phase,opt.dataroot,opt.category,opt.n_pts)
    dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=opt.batch_size,
                                                shuffle=True,
                                                num_workers=opt.nThreads)

    val_dataset = PartNetdataset("validation",opt.dataroot,opt.category,opt.n_pts)
    val_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=opt.batch_size,
                                                shuffle=False,
                                                num_workers=opt.nThreads)
    

    if opt.phase == "train":
        now = (datetime.utcnow()+timedelta(hours=8)).isoformat()

        opt.save_path = os.path.join('./log', 'PA1/train', (opt.category+now))
        if not os.path.exists(opt.save_path):
            os.makedirs(opt.save_path)
        # os.system('cp ./models/latent_3d_points/l-gan.py %s' % opt.save_path)

        print('------------ Options -------------')
        for k, v in sorted(vars(opt).items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        with open(os.path.join(opt.save_path, 'runlog.txt'), 'a') as f:
            f.write('------------ Options -------------\n')
            for k, v in sorted(vars(opt).items()):
                f.write('%s: %s\n' % (str(k), str(v)))
            f.write('-------------- End ----------------\n')

        runner = Runner(opt)
        for epoch in range(1, opt.n_epochs+1):
            for i, data in enumerate(dataloader):
                if i>2:
                    break
                runner.train_func(data=data, args=[epoch, i+1, len(dataloader)])
            if epoch % 50 == 0:
                for i, data in enumerate(val_dataloader):
                    runner.val_func(data=data)
                runner.val_model(args=[epoch])
            if epoch > 200:
                runner.after_one_epoch(args=[epoch])

    if opt.mode == "test":
        test_dataset = PartNetdataset("validation",opt.dataroot,opt.category,opt.n_pts)
        test_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=opt.batch_size,
                                                    shuffle=False,
                                                    num_workers=opt.nThreads)
        opt.save_path = opt.ckpt_path[::-1].split('/', 3)[-1][::-1]

        opt.save_pc_path = os.path.join('./log', 'PA1/test', opt.category)
        runner = Runner(opt)
        runner.load_pretrained()
        
        for i, data in enumerate(test_dataloader):
            runner.test_func(data=data)
            