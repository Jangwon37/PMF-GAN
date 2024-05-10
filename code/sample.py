import os
import sys

# Setting up environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append(os.getcwd())

from utils.utility import *
from models.loss_model import Euclidean, Chi2, Chybyshev, Manhattan, SquaredChord
from data.data_loader import DataLoad

import argparse
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision.models import inception_v3

# Setting up CUDA
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Tensor = torch.cuda.FloatTensor if DEVICE.type == "cuda" else torch.FloatTensor

# Inception model setting
inception_model = inception_v3(pretrained=True, transform_input=False)
inception_model.fc = torch.nn.Identity()
inception_model = inception_model.to(DEVICE)

loss_function_name = [
    "WGAN-GP",
    "LSGAN-GP",
    "SphereGAN",
    "Euclidean",
    "Chi2",
    "Chybyshev",
    "Manhattan",
    "SquaredChord",
]

# mnist, cifar // epoch: 256, batch size: 128
# celeba // epoch: 64, batch size: 128
# lsun // epoch: 43, batch size 128
# afhq // epoch: 437, batch size 64

parser = argparse.ArgumentParser()
parser.add_argument("--n_bin", type=int, default=3, help="histogram's bin")
parser.add_argument('--dataset', default='afhq', help='Enter the dataset you want the model to train on')
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--Glr", type=float, default=2, help="G: learning rate")
parser.add_argument("--Dlr", type=float, default=2, help="D: learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--top_is", type=bool, default=False, help="Training load when top_is is large")
parser.add_argument("--mode", default="client")
parser.add_argument("--port", default=58614)

def main():
    opt = parser.parse_args()
    print(opt)

    for i, name in enumerate(loss_function_name):
        print(f'{i} : {name}')
    print("9 : all model")

    model_select = int(input("loss number :  "))

    dataset_info = {
        'mnist': (25, 1),
        'f_mnist': (25, 1),
        'cifar10': (64, 3),
        'cifar100': (64, 3),
        'celeba': (64, 3),
        'lsun': (64, 3),
        'afhq': (64, 3)
    }
    n_image, channel = dataset_info.get(opt.dataset, (None, None))

    fix_z = Variable(Tensor(np.random.normal(0, 1, (n_image, opt.latent_dim))).to(DEVICE))

    if opt.dataset in ['cifar10', 'cifar100']:
        batch_size = 128
    elif opt.dataset in ['celeba']:
        batch_size = 128
    elif opt.dataset in ['lsun']:
        batch_size = 128
    else:
        batch_size = 64

    loop_start = 100
    all_model = 108
    for model_loop in range(loop_start, all_model, 1):
        rmb = -1
        if model_select == 9:
            rmb = 1
            model_select = model_loop - 100

        # image save path
        savepath = createFolder(
            f"../result/H{opt.n_bin}_B{batch_size}_lr{opt.Glr}_{opt.Dlr}/{opt.dataset}_images/{loss_function_name[model_select]}/")
        png_path = createFolder(f'{savepath}png/')

        # Loss function
        loss_function = [Euclidean, Chi2, Chybyshev, Manhattan, SquaredChord]

        if model_select == 0:
            print('wgan-GP')
        elif model_select == 1:
            print('lsgan-GP')
        elif model_select == 2:
            print('SphereGAN')
        else:
            print(loss_function[model_select - 3])

        # load model
        if opt.top_is == True:
            generator = torch.load(savepath + "G").to(DEVICE)
        else:
            generator = torch.load(savepath + "G_").to(DEVICE)

        generator.eval()
        gen_imgs_fix = generator(fix_z)
        save_image(gen_imgs_fix.data[:9], f"{png_path}/{loss_function_name[model_select]}.tiff", nrow=3, normalize=True, dpi=300)
        print(loss_function_name[model_select])

        if rmb == 1:
            model_select = 9
            model_loop = 1

        if model_loop == loop_start:
            break

if __name__ == "__main__":
    main()