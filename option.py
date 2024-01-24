import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=str, default='cuda:0')

parser.add_argument('--epochs', type=int, default=2000, help='maximum number of epochs to train the total model.')
parser.add_argument('--epochs_encoder', type=int, default=200, help='number of epochs to train encoder.')
parser.add_argument('--epochs_select', type=int, default=20, help='number of epochs to train encoder.')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate of encoder.')

parser.add_argument('--de_type', type=list, default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze'],
                    help='which type of degradations is training and testing for.')

parser.add_argument('--patch_size', type=int, default=128, help='patcphsize of input.')
parser.add_argument('--encoder_dim', type=int, default=256, help='the dimensionality of encoder.')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers.')

# path
parser.add_argument('--data_file_dir', type=str, default='data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--denoise_dir', type=str, default='data/Train/Denoise/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='data/Train/Derain/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='data/Train/Dehaze/',
                    help='where training images of dehazing saves.')
parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/slimmable_5types_distillation/", help='checkpoint save path')
parser.add_argument('--stage', type=int, default=2, help='checkpoint save path')
parser.add_argument('--target', type=float, default=0.8, help='target poportion')
options = parser.parse_args()
options.batch_size = len(options.de_type)
