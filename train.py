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
    net = nn.DataParallel(AirNet(opt, width_mult_list=width_mult_list)).cuda()
    net.train()

    # Optimizer and Loss
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    CE = nn.CrossEntropyLoss().cuda()
    l1 = nn.L1Loss().cuda()

    # Start training
    print('Start training...')
    for epoch in range(opt.epochs):
        if epoch <= opt.epochs_encoder:
            lr = opt.lr * (0.1 ** (epoch // 120))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = 0.0001 * (0.5 ** ((epoch - opt.epochs_encoder) // 250))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        for ([clean_name, de_id], degrad_patch_1, degrad_patch_2, clean_patch_1, clean_patch_2) in tqdm(trainloader):
            degrad_patch_1, degrad_patch_2 = degrad_patch_1.cuda(), degrad_patch_2.cuda()
            clean_patch_1, clean_patch_2 = clean_patch_1.cuda(), clean_patch_2.cuda()

            net.module.set_slimmable_ratio(width_mult_list[int(np.random.randint(0, len(width_mult_list) - 1))])
            optimizer.zero_grad()

            if epoch < 0:
                _, output, _ = net(x_query=degrad_patch_1, x_key=degrad_patch_2, forward_E=True)
                contrast_loss = CE(output, torch.LongTensor(de_id).cuda())
                loss = contrast_loss
            else:
                net.module.set_slimmable_ratio(width_mult_list[int(np.random.randint(0, len(width_mult_list) - 1))])
                restored, output = net(x_query=degrad_patch_1, x_key=degrad_patch_2)
                contrast_loss = CE(output, torch.LongTensor(de_id).cuda())
                net.module.set_slimmable_ratio(1.0)
                restored_full, _ = net(x_query=degrad_patch_1, x_key=degrad_patch_2)
                l1_loss = l1(restored, clean_patch_1)
                l1_loss_full = l1(restored_full, clean_patch_1)
                l1_loss_distillation = l1(restored, restored_full.detach())
                loss = l1_loss + 5 * l1_loss_full + 0.1 * contrast_loss

            # backward
            loss.backward()
            optimizer.step()

        if epoch < opt.epochs_encoder:
            print(
                'Epoch (%d)  Loss: contrast_loss:%0.4f\n' % (
                    epoch, contrast_loss.item(),
                ), '\r', end='')
        else:
            print(
                'Epoch (%d)  Loss: l1_loss:%0.4f l1_loss_full:%0.4f contrast_loss:%0.4f\n' % (
                    epoch, l1_loss.item(), l1_loss_full.item(), contrast_loss.item(),
                ), '\r', end='')

        GPUS = 1
        if (epoch + 1) % 50 == 0:
            checkpoint = {
                "net": net.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch
            }
            if GPUS == 1:
                torch.save(net.state_dict(), opt.ckpt_path + 'epoch_' + str(epoch + 1) + '.pth')
            else:
                torch.save(net.module.state_dict(), opt.ckpt_path + 'epoch_' + str(epoch + 1) + '.pth')

        
