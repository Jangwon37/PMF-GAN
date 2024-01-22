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
    "WGAN",
    "LSGAN",
    "Euclidean",
    "Chi2",
    "Chybyshev",
    "Manhattan",
    "SquaredChord",
]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=256, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--n_bin", type=int, default=3, help="histogram's bin")
    parser.add_argument('--dataset', default='celeba', help='Enter the dataset you want the model to train on')
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--Glr", type=float, default=2, help="G: learning rate")
    parser.add_argument("--Dlr", type=float, default=2, help="D: learning rate")
    parser.add_argument("--b1", type=float, default=0., help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
    parser.add_argument("--sample_interval", type=int, default=5000, help="interval between image samples")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--mode", default="client")
    parser.add_argument("--port", default=58614)
    parser.add_argument("--loop", default=1)
    return parser.parse_args(args=[])


opt = get_args()
print(opt)

def load_data(use_data):
    loaders = {
        'mnist': DataLoad().load_data_mnist,
        'f_mnist': DataLoad().load_data_f_mnist,
        'cifar10': DataLoad().load_data_cifar10,
        'cifar100': DataLoad().load_data_cifar100,
        'celeba': DataLoad().load_data_celeba,
        'lsun': DataLoad().load_data_lsun
    }
    return loaders.get(use_data, lambda: print('train loader error'))(batch_size=opt.batch_size)

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
    'lsun': (64, 3)
}
n_image, channel = dataset_info.get(opt.dataset, (None, None))

fix_z = Variable(Tensor(np.random.normal(0, 1, (n_image, opt.latent_dim))).to(DEVICE))

loop_start = 100
all_model = 107
for model_loop in range(loop_start, all_model, 1):
    rmb = -1
    if model_select == 9:
        rmb = 1
        model_select = model_loop - 100

    # image save path
    savepath = createFolder(
        f"../result/H{opt.n_bin}_B{opt.batch_size}_lr{opt.Glr}_{opt.Dlr}/{opt.dataset}_images/{loss_function_name[model_select]}/")
    png_path = createFolder(f'{savepath}png/')

    train_loader = load_data(opt.dataset)

    if opt.dataset in ['cifar10', 'cifar100', 'celeba', 'lsun']:
        real_imgs_fid = createFolder(f'../../../data/sample/{opt.dataset}')
        save_samples_from_loader(train_loader, real_imgs_fid)

    # Loss function
    loss_function = [Euclidean, Chi2, Chybyshev, Manhattan, SquaredChord]

    if model_select == 0:
        print('wgan')
    elif model_select == 1:
        print('lsgan')
    else:
        adversarial_loss = loss_function[model_select - 2](opt.batch_size, opt.n_bin).to(DEVICE)
        print(loss_function[model_select - 2])

    # load model
    generator = torch.load(savepath + "G_").to(DEVICE)
    discriminator = torch.load(savepath + "D_").to(DEVICE)

    generator.eval()
    gen_imgs_fix = generator(fix_z)
    save_image(gen_imgs_fix.data[:9], f"{png_path}/{loss_function_name[model_select]}.tiff", nrow=3, normalize=True, dpi=300)
    print(loss_function_name[model_select])

    if rmb == 1:
        model_select = 9
        model_loop = 1

    if model_loop == loop_start:
        break