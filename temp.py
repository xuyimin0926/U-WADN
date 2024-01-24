import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor

from net.model import AirNet

def test_Denoise(net, dataset, sigma=15):
    output_path = opt.output_path + 'denoise/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(x_query=degrad_patch, x_key=degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + clean_name[0] + '.png')

        print("Deonise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))


parser = argparse.ArgumentParser()

parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--mode', type=int, default=0,
                    help='0 for denoise, 1 for derain, 2 for dehaze, 3 for all-in-one')

parser.add_argument('--denoise_path', type=str, default="airnet_test_data/denoise/BSD68/", help='save path of test noisy images')
parser.add_argument('--derain_path', type=str, default="airnet_test_data/derain/", help='save path of test raining images')
parser.add_argument('--dehaze_path', type=str, default="airnet_test_data/dehaze/", help='save path of test hazy images')
parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/slimmable_3types_selector_0.01lr/", help='checkpoint save path')
parser.add_argument('--slimmable_ratio', type=float, default=0.2)
parser.add_argument('--stage', type=int, default=2)
opt = parser.parse_args()

np.random.seed(0)
torch.manual_seed(0)

opt.stage = 1
opt.batch_size=5
width_mult_list = [0.5, 0.75, 1.0]
net = AirNet(opt, width_mult_list=width_mult_list).cuda()
ckpt_path = "ckpt/slimmable_3types/epoch_2000.pth"
state_dict = torch.load(ckpt_path, map_location=torch.device(opt.cuda))
fix_state_dict = {}
for name in state_dict.keys():
    if 'R.' in name or 'pre' in name or 'transform' in name:
        fix_state_dict[name] = state_dict[name]
    else:
        print(name + '\n')
net.load_state_dict(fix_state_dict, strict=False)
net.eval()
net.set_slimmable_ratio(1.0)

denoise_set = DenoiseTestDataset(opt)

print('Start testing Sigma=15...')
test_Denoise(net, denoise_set, sigma=15)