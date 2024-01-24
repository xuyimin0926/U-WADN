import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
import time
from torch.utils.data import DataLoader

from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor

from net.model import AirNet
from thop import profile


def test_Denoise(net, dataset, sigma=15):
    output_path = opt.output_path + 'denoise/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    psnr = AverageMeter()
    ssim = AverageMeter()
    selection = AverageMeter()
    t = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
            start.record()
            restored, s = net(x_query=degrad_patch, x_key=degrad_patch)
            end.record()
            torch.cuda.synchronize()
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            t.update(start.elapsed_time(end), N)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            selection.update(s, N)

            save_image_tensor(restored, output_path + clean_name[0] + '.png')
        
        flops, params = profile(net, inputs=(degrad_patch,degrad_patch,), verbose=True)
        print('FLOPs = ' + str(flops/1000**3) + 'G')
        print('Params = ' + str(params/1000**2) + 'M')

        print("Deonise sigma=%d: psnr: %.2f, ssim: %.4f, usage:%.4f, time:%.4f" % (sigma, psnr.avg, ssim.avg, selection.avg,  t.avg))


def test_Derain_Dehaze(net, dataset, task="derain"):
    output_path = opt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()
    selection = AverageMeter()
    t = AverageMeter()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            start.record()
            restored, s = net(x_query=degrad_patch, x_key=degrad_patch)
            end.record()
            torch.cuda.synchronize()
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            t.update(start.elapsed_time(end), N)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            selection.update(s, N)

            save_image_tensor(restored, output_path + degraded_name[0] + '.png')

        print("PSNR: %.2f, SSIM: %.4f, usage:%.4f, time:%.4f" % (psnr.avg, ssim.avg, selection.avg, t.avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=0,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for all-in-one')

    parser.add_argument('--denoise_path', type=str, default="airnet_test_data/denoise/BSD68/", help='save path of test noisy images')
    parser.add_argument('--derain_path', type=str, default="airnet_test_data/derain/", help='save path of test raining images')
    parser.add_argument('--dehaze_path', type=str, default="airnet_test_data/dehaze/", help='save path of test hazy images')
    parser.add_argument('--output_path', type=str, default="output/", help='output save path')
    parser.add_argument('--ckpt_path', type=str, default="ckpt/ckpt_best/", help='checkpoint save path')
    parser.add_argument('--slimmable_ratio', type=float, default=0.2)
    parser.add_argument('--stage', type=int, default=2)
    opt = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(opt.cuda)

    if opt.mode == 0:
        opt.batch_size = 3
        ckpt_path = opt.ckpt_path + 'best.pth'
    elif opt.mode == 1:
        opt.batch_size = 1
        ckpt_path = opt.ckpt_path + 'best.pth'
    elif opt.mode == 2:
        opt.batch_size = 1
        ckpt_path = opt.ckpt_path + 'best.pth'
    elif opt.mode == 3:
        opt.batch_size = 5
        ckpt_path = opt.ckpt_path + 'best.pth'

    denoise_set = DenoiseTestDataset(opt)
    derain_set = DerainDehazeDataset(opt)

    # Make network
    width_mult_list = [0.6, 0.7, 0.8, 0.9, 1.0]
    net = AirNet(opt, width_mult_list=width_mult_list).cuda()
    # assert opt.slimmable_ratio in [0.6, 0.7, 0.8, 0.9, 1.0]
    # net.set_slimmable_ratio(opt.slimmable_ratio)
    net.eval()
    net.load_state_dict(torch.load(ckpt_path, map_location=torch.device(opt.cuda)))

    if opt.mode == 0:
        print('Start testing Sigma=15...')
        test_Denoise(net, denoise_set, sigma=15)

        print('Start testing Sigma=25...')
        test_Denoise(net, denoise_set, sigma=25)

        print('Start testing Sigma=50...')
        test_Denoise(net, denoise_set, sigma=50)
    elif opt.mode == 1:
        print('Start testing rain streak removal...')
        test_Derain_Dehaze(net, derain_set, task="Rain100L")
    elif opt.mode == 2:
        print('Start testing SOTS...')
        test_Derain_Dehaze(net, derain_set, task="SOTS_outdoor")
    elif opt.mode == 3:
        print('Start testing Sigma=15...')
        test_Denoise(net, denoise_set, sigma=15)

        print('Start testing Sigma=25...')
        test_Denoise(net, denoise_set, sigma=25)

        print('Start testing Sigma=50...')
        test_Denoise(net, denoise_set, sigma=50)

        print('Start testing rain streak removal...')
        test_Derain_Dehaze(net, derain_set, task="derain")

        print('Start testing SOTS...')
        test_Derain_Dehaze(net, derain_set, task="dehaze")
