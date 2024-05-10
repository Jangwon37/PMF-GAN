import os
import sys

# Setting up environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append(os.getcwd())

import gc
import time
import shutil
import argparse
from tqdm import tqdm
from data.data_loader import DataLoad
from models.loss_model import Euclidean, Chi2, Chybyshev, Manhattan, SquaredChord
from models.model import Generator, Discriminator
from utils.score import *
from utils.utility import *

import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision.models import inception_v3
from pytorch_fid.fid_score import calculate_fid_given_paths

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
parser.add_argument("--n_epochs", type=int, default=437, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--n_bin", type=int, default=3, help="histogram's bin")
parser.add_argument('--dataset', default='afhq', help='Enter the dataset you want the model to train on')
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--Glr", type=float, default=2, help="G: learning rate")
parser.add_argument("--Dlr", type=float, default=2, help="D: learning rate")
parser.add_argument("--b1", type=float, default=0., help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--sample_interval", type=int, default=5000, help="interval between image samples")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--top_is", type=bool, default=False, help="Training load when top_is is large")
parser.add_argument("--mode", default="client")
parser.add_argument("--port", default=58614)
parser.add_argument("--loop", type=int, default=1)

def main():
    opt = parser.parse_args()
    print(opt)

    def load_data(use_data):
        loaders = {
            'mnist': DataLoad().load_data_mnist,
            'f_mnist': DataLoad().load_data_f_mnist,
            'cifar10': DataLoad().load_data_cifar10,
            'cifar100': DataLoad().load_data_cifar100,
            'celeba': DataLoad().load_data_celeba,
            'lsun': DataLoad().load_data_lsun,
            'afhq': DataLoad().load_data_afhq
        }
        return loaders.get(use_data, lambda: print('train loader error'))(batch_size=opt.batch_size)

    for i, name in enumerate(loss_function_name):
        print(f'{i} : {name}')
    print("9 : all model")

    model_select = int(input("loss number :  "))

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
        is_path = f"{savepath}IS.csv"
        is_epoch_path = f"{savepath}IS_epoch.csv"
        batches_done_path = f"{savepath}batches_done.txt"
        batches_done_distribution_path = f"{savepath}batches_done_distribution.txt"


        train_loader = load_data(opt.dataset)

        if opt.dataset in ['cifar10', 'cifar100', 'celeba', 'lsun', 'afhq']:
            real_imgs_fid = createFolder(f'../../../data/sample/{opt.dataset}')
            save_samples_from_loader(train_loader, real_imgs_fid)

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

        # ShowImage(train_loader, n_image)

        # Loss weight for gradient penalty
        lambda_gp = 10

        # Loss function
        loss_function = [Euclidean, Chi2, Chybyshev, Manhattan, SquaredChord]

        if model_select == 0:
            print('WGAN-GP')
        elif model_select == 1:
            print('LSGAN-GP')
        else:
            adversarial_loss = loss_function[model_select - 2](opt.batch_size, opt.n_bin).to(DEVICE)
            print(loss_function[model_select - 2])

        if os.path.isfile(is_epoch_path):
            data_count = pd.read_csv(is_epoch_path, header=None).shape[0]
        else:
            data_count = 0

        if data_count == opt.n_epochs:
            d_loss = torch.tensor(1)
            # load model
            if opt.top_is == True:
                generator = torch.load(savepath + "G").to(DEVICE)
                discriminator = torch.load(savepath + "D").to(DEVICE)
            else:
                generator = torch.load(savepath + "G_").to(DEVICE)
                discriminator = torch.load(savepath + "D_").to(DEVICE)

        else:
            if (os.path.isfile(savepath + "G_") and os.path.isfile(savepath + "D_")) == True:
                generator = torch.load(savepath + "G_").to(DEVICE)
                discriminator = torch.load(savepath + "D_").to(DEVICE)

            else:
                # Initialize generator and discriminator
                generator = Generator(opt.dataset, opt.latent_dim).to(DEVICE)
                discriminator = Discriminator(opt.dataset).to(DEVICE)

            IS_data_b = 0
            start_time = time.time()

            # Optimizers
            optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr * opt.Glr, betas=(opt.b1, opt.b2))
            optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr * opt.Dlr, betas=(opt.b1, opt.b2))

            np.random.seed(0)

            fix_z = Variable(Tensor(np.random.normal(0, 1, (n_image, opt.latent_dim))).to(DEVICE))

            # ----------
            #  Training
            # ----------

            for epoch in range(data_count, opt.n_epochs):
                for i, (imgs, _) in enumerate(train_loader):

                    if os.path.isfile(batches_done_path):
                        with open(batches_done_path, 'r') as f:
                            batches_done = int(f.read().strip())
                    else:
                        batches_done = 0

                    if os.path.isfile(batches_done_distribution_path):
                        with open(batches_done_distribution_path, 'r') as f:
                            batches_done_distribution = int(f.read().strip())
                    else:
                        batches_done_distribution = 0

                    # Configure input
                    real_imgs = Variable(imgs.type(Tensor).to(DEVICE))

                    # ---------------------
                    #  Train Discriminator
                    # ---------------------

                    optimizer_D.zero_grad()

                    z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))).to(DEVICE))

                    generator.to(DEVICE)
                    generator.eval()

                    gc.collect()
                    torch.cuda.empty_cache()

                    gen_imgs = generator(z).detach()

                    if model_select == 0:  # WGAN-GP
                        fake_loss = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(gen_imgs))
                    elif model_select == 1:  # LSGAN-GP
                        real_loss = torch.nn.MSELoss()(discriminator(real_imgs), torch.ones_like(discriminator(real_imgs)))
                        fake_loss_1 = torch.nn.MSELoss()(discriminator(gen_imgs), torch.zeros_like(discriminator(gen_imgs)))
                        fake_loss = (real_loss + fake_loss_1) / 2
                    else:
                        fake_loss = -1. * adversarial_loss(discriminator(gen_imgs), discriminator(real_imgs))

                    # Gradient penalty
                    gradient_penalty = lambda_gp * compute_gradient_penalty(discriminator, real_imgs.data, gen_imgs.data)
                    d_loss = fake_loss + gradient_penalty
                    d_loss.backward()

                    optimizer_D.step()

                    optimizer_G.zero_grad()

                    if i % opt.n_critic == 0:

                        # -----------------
                        #  Train Generator
                        # -----------------

                        # Sample noise as generator input
                        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))).to(DEVICE))
                        generator.train()
                        # Generate a batch of images
                        gc.collect()
                        torch.cuda.empty_cache()

                        gen_imgs = generator(z)

                        # Loss measures generator's ability to fool the discriminator
                        if model_select == 0:  # WGAN-GP
                            g_loss = -torch.mean(discriminator(gen_imgs))
                        elif model_select == 1:  # LSGAN-GP
                            g_loss = torch.nn.MSELoss()(discriminator(gen_imgs), torch.ones_like(discriminator(gen_imgs)))
                        else:
                            g_loss = adversarial_loss(discriminator(gen_imgs), discriminator(real_imgs))

                        g_loss.backward()
                        optimizer_G.step()

                        print(
                            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                            % (epoch, opt.n_epochs, i, len(train_loader), d_loss.item(), g_loss.item())
                        )

                        if batches_done % opt.sample_interval == 0:
                            generator.eval()
                            gen_imgs_fix = generator(fix_z)
                            save_image(gen_imgs_fix.data[:n_image], png_path + "%d.png" % batches_done,
                                       nrow=int(np.sqrt(n_image)), normalize=True)

                        batches_done += opt.n_critic

                        with open(batches_done_path, 'w') as f:
                            f.write(str(batches_done))

                    if torch.isnan(d_loss):
                        break

                    # save loss
                    g_loss_path = f"{savepath}/g_loss.csv"
                    d_loss_path = f"{savepath}/d_loss.csv"
                    write_mode = 'a' if os.path.isfile(g_loss_path) else 'w'

                    pd.DataFrame([g_loss.item()]).to_csv(g_loss_path, mode=write_mode, header=False, index=False)
                    pd.DataFrame([d_loss.item()]).to_csv(d_loss_path, mode=write_mode, header=False, index=False)

                    if model_select == 0 or model_select == 1:
                        if batches_done_distribution % 500 == 0:
                            D_responses = createFolder(f'{savepath}/responses/')
                            D_responses_plot = createFolder(f'{D_responses}/plot/')
                            D_responses_array = createFolder(f'{D_responses}/array/')
                            plt.clf()

                            with torch.no_grad():
                                discriminator_responses_on_generated = discriminator(gen_imgs).cpu().numpy()
                                discriminator_responses_on_real = discriminator(real_imgs).cpu().numpy()

                            save_discriminator_responses(batches_done_distribution, discriminator_responses_on_generated, discriminator_responses_on_real, D_responses_array)
                            plt.hist(discriminator_responses_on_generated, bins=40, density=True, alpha=0.5, label='Generated')
                            plt.hist(discriminator_responses_on_real, bins=40, density=True, alpha=0.5, label='Real')
                            plt.xlabel('Discriminator Responses')
                            plt.ylabel('Likelihood')
                            plt.legend()
                            plt.savefig(f'{D_responses_plot}{batches_done_distribution}.png')
                            plt.clf()
                        batches_done_distribution += 1

                        with open(batches_done_distribution_path, 'w') as f:
                            f.write(str(batches_done_distribution))

                if torch.isnan(d_loss):
                    break

                # save IS
                if opt.dataset in ['cifar10', 'cifar100', 'celeba', 'lsun', 'afhq']:

                    num_images_per_batch = 100
                    num_batches = 50

                    total_features = np.zeros((num_images_per_batch * num_batches, 1000))
                    for num_batches_i in tqdm(range(num_batches)):
                        z = Variable(Tensor(np.random.normal(0, 1, (num_images_per_batch, opt.latent_dim))).to(DEVICE))
                        gen_imgs = generator(z).to(DEVICE)
                        total_features[num_batches_i * num_images_per_batch:(num_batches_i + 1) * num_images_per_batch] = get_feature_vectors(gen_imgs, resize=True)

                    print("Calculating Inception Score...")
                    IS_data = calculate_inception_score(total_features)
                    print("IS Score:", IS_data)

                    if IS_data_b < IS_data:
                        IS_data_b = IS_data

                        # Save model
                        torch.save(generator, savepath + "G")
                        torch.save(discriminator, savepath + "D")

                    # Handle CSV writing
                    write_mode = 'a' if os.path.isfile(is_epoch_path) else 'w'
                    pd.DataFrame([IS_data]).iloc[0].to_csv(is_epoch_path, mode=write_mode, header=False, index=False)

            generator.eval()
            discriminator.eval()

            # save final model
            torch.save(generator, savepath + "G_")
            torch.save(discriminator, savepath + "D_")

            # save time
            elapsed_time = time.time() - start_time
            print(f"--- {elapsed_time} seconds ---")
            pd.DataFrame([elapsed_time]).to_csv(f'{savepath}time.csv', header=False, index=False)

            # loss graph
            G_losses = pd.read_csv(f"{savepath}g_loss.csv")
            D_losses = pd.read_csv(f"{savepath}d_loss.csv")

            plt.figure(figsize=(10, 5))
            plt.title("Generator and Discriminator Loss During Training")
            plt.plot(G_losses, label="G")
            plt.plot(D_losses, label="D")
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(f'{savepath}GD_loss.tiff', dpi=300)

            # Clearing cache
            gc.collect()
            torch.cuda.empty_cache()

        if opt.dataset in ['cifar10', 'cifar100', 'celeba', 'lsun', 'afhq']:
            for n_loop in range(opt.loop):
                if torch.isnan(d_loss):
                    break

                num_images_per_batch = 100
                num_batches = 50

                total_features = np.zeros((num_images_per_batch * num_batches, 1000))
                for num_batches_i in tqdm(range(num_batches)):
                    z = Variable(Tensor(np.random.normal(0, 1, (num_images_per_batch, opt.latent_dim))).to(DEVICE))
                    gen_imgs = generator(z).to(DEVICE)
                    total_features[num_batches_i * num_images_per_batch:(num_batches_i + 1) * num_images_per_batch] = get_feature_vectors(gen_imgs, resize=True)

                print("Calculating Inception Score...")
                IS_data = calculate_inception_score(total_features)
                print("IS Score:", IS_data)

                write_mode = 'a' if os.path.isfile(is_path) else 'w'
                pd.DataFrame([IS_data]).iloc[0].to_csv(is_path, mode=write_mode, header=False, index=False)

                print("Calculating FID Score...")

                gen_imgs_fid = createFolder(f"{savepath}/gen_img/")

                save_generated_images(generator, opt.latent_dim, gen_imgs_fid)

                path_fid = [gen_imgs_fid, real_imgs_fid]

                fid_score = calculate_fid_given_paths(path_fid,
                                                      batch_size=opt.batch_size,
                                                      device=DEVICE,
                                                      dims=2048)
                print("FID Score:", fid_score)

                fid_path = f"{savepath}FID.csv"
                write_mode = 'a' if os.path.isfile(fid_path) else 'w'
                pd.DataFrame([fid_score]).to_csv(fid_path, mode=write_mode, header=False, index=False)

                if os.path.exists(gen_imgs_fid):
                    shutil.rmtree(gen_imgs_fid)
                    print("The image folder has been successfully deleted.")
                else:
                    print("The image folder does not exist.")

        print(loss_function_name[model_select])

        if rmb == 1:
            model_select = 9
            model_loop = 1

        if model_loop == loop_start:
            break

if __name__ == "__main__":
    main()