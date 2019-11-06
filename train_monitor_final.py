import argparse
import random
import math

from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from dataset import MultiResolutionDataset
from model import StyledGenerator, Discriminator

import random
import time
import paths
import os

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(dataset, batch_size, image_size=4):
    dataset.resolution = image_size
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=32)

    return loader


def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult


def train_monitorExp(model, resolution, batch_size, label_size):
    requires_grad(model, True)
    step = int(math.log2(resolution)) - 2
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.0, 0.99), weight_decay=1e-5)
    L1loss = nn.L1Loss()
    MSEloss = nn.MSELoss()
    CEloss = nn.CrossEntropyLoss()
    
    dataset = MultiResolutionDataset(resolution, exclude_neutral=True)
    
    data_loader = iter(DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=16))
        
    pbar = tqdm(range(20_000))
    for i in pbar:        
        
        tick = time.time()
        img, label = next(data_loader)
        tock = time.time()

        image = img.cuda()
        target = label.cuda().float()
        predict = model(image, step=step, alpha=1.0)
        
        model.zero_grad()

        loss = MSEloss(predict, target) 

        loss.backward()
        optimizer.step()
            
        state_msg = (
            f'[MonitorExp] Size: {4 * 2 ** step}; MSE_loss: {loss.item():.3f}; Data: {tock-tick:.3f};'
        )

        pbar.set_description(state_msg)
        
        if i%200 == 0:
            np.set_printoptions(precision=2, suppress=True)
            print (' ----------------------------- ')
            print (predict.data.cpu().numpy()[0])
            print (target.data.cpu().numpy()[0])
        
        if i%1000 == 0:
            torch.save(
                {
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                f'checkpoint/monitorExp/resolution-{resolution}-iter-{i}.model',
            )

    torch.save(
        {
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        },
        f'checkpoint/monitorExp/resolution-{resolution}-iter-{i}.model',
    )
    requires_grad(model, False)
    return model



        
# python trainxxx.py --trainExp 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Aux monitors')
    parser.add_argument('--trainExp', action='store_true')
    parser.add_argument('--testExp', action='store_true')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
    
    args.batch = {64: 32, 128: 16, 256: 8, 512: 4, 1024: 2}

    label_size = 17*3
    
    if args.trainExp:
        for step in [8]:
            resolution = 2 ** step
            monitorExp = nn.DataParallel(Discriminator(from_rgb_activate=True, out_channel=label_size)).cuda()
            # ckpt = torch.load(f'checkpoint/monitorExp/resolution-{2 ** step}-iter-{19999}.model')
            # monitorExp.module.load_state_dict(ckpt['model'])

            # pretrained_dict = torch.load(f'checkpoint/monitorExp/resolution-{2 ** step}-iter-{3000}.model')['model']
            # model_dict = monitorExp.module.state_dict()
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # model_dict.update(pretrained_dict)
            # monitorExp.module.load_state_dict(model_dict)

            batch_size = args.batch.get(resolution, 32) * 30
            monitorID = train_monitorExp(monitorExp, resolution, batch_size, label_size)
            torch.cuda.empty_cache()