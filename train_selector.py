import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from utils.dataset_utils import TrainDataset, TrainSampler
from net.model import AirNet

from option import options as opt
import os

if __name__ == '__main__':
    subprocess.check_output(['mkdir', '-p', opt.ckpt_path])

    trainset = TrainDataset(opt)
    train_sampler = TrainSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True,
                             drop_last=True, num_workers=opt.num_workers, sampler=train_sampler)

    width_mult_list = [0.6, 0.7, 0.8, 0.9, 1.0]
    # Network Construction
    net = AirNet(opt, width_mult_list=width_mult_list).cuda()
    ckpt_path = "ckpt/235ckpt/epoch_2000.pth"
    state_dict = torch.load(ckpt_path, map_location=torch.device(opt.cuda))
    fix_state_dict = {}
    for name in state_dict.keys():
        if 'R.' in name or 'pre' in name or 'transform' in name:
            fix_state_dict[name] = state_dict[name]
    net.load_state_dict(fix_state_dict, strict=False)
    net.train()
    # net.module.fix_gradient()
    grad_m = []
    for n, m in net.named_parameters():
        if 'R.' in n or 'pre' in n or 'transform' in n:
            m.requires_grad_(False)
        else:
            grad_m.append(m)
            print(n)
    
    for n, m in net.named_modules():
        if 'R.' in n or 'pre' in n or 'transform' in n:
            m.eval()
    # print(grad_m)
    # Optimizer and Loss
    optimizer = optim.Adam(grad_m, lr=0.01)
    schduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs_select)
    CE = nn.CrossEntropyLoss().cuda()
    l1 = nn.L1Loss(reduction='none').cuda()

    # Start training
    print('Start training...')
    for epoch in range(opt.epochs_select):
        for ([clean_name, de_id], degrad_patch_1, degrad_patch_2, clean_patch_1, clean_patch_2) in tqdm(trainloader):
            degrad_patch_1, degrad_patch_2 = degrad_patch_1.cuda(), degrad_patch_2.cuda()
            clean_patch_1, clean_patch_2 = clean_patch_1.cuda(), clean_patch_2.cuda()

            optimizer.zero_grad()
            
            select_loss = torch.tensor(0.).cuda()

            output, logits, final = net(x_query=degrad_patch_1, x_key=degrad_patch_2, forward_E=False)
            final = torch.softmax(final, dim=1)
            for i, width_multi in enumerate(width_mult_list):
                with torch.no_grad():
                    net.set_slimmable_ratio(width_multi)
                    restored, _, _ = net(x_query=degrad_patch_1, x_key=degrad_patch_2, forward_E=False)
                select_loss += torch.mean((final[:, i] * l1(restored, clean_patch_1).mean((1,2,3))))
            contrast_loss = CE(logits, torch.LongTensor(de_id).cuda())
            sparse_loss = (torch.sum(final * torch.tensor(width_mult_list).cuda())/len(de_id) - opt.target) ** 2
            loss = contrast_loss + select_loss + sparse_loss

            # backward
            loss.backward()
            optimizer.step()
            schduler.step()

        print(
            'Epoch (%d)  Loss: contrast_loss:%0.4f select_loss:%0.4f sparse_loss:%0.4f\n' % (
                epoch, contrast_loss.item(), select_loss.item(), sparse_loss.item(),
            ), '\r', end='')

        GPUS = 1 
        if (epoch + 1) % 1 == 0:
            checkpoint = {
                "net": net.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch
            }

            if GPUS == 1:
                torch.save(net.state_dict(), opt.ckpt_path + 'epoch_' + str(epoch + 1) + '.pth')
            else:
                torch.save(net.module.state_dict(), opt.ckpt_path + 'epoch_' + str(epoch + 1) + '.pth')
