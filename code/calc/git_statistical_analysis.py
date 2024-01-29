import os
from typing import Tuple, List, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Function to load data from CSV files
def load_data(file_path):
    return np.array(pd.read_csv(file_path, header=None))

# Function to process IS and FID data for a single model
def process_model_data(model_path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Any], List[Any]]:
    IS_epoch_path = os.path.join(model_path, 'IS_epoch.csv')
    IS_path = os.path.join(model_path, 'IS.csv')
    FID_path = os.path.join(model_path, 'FID.csv')

    IS_epoch_data = load_data(IS_epoch_path)
    IS = load_data(IS_path)
    FID = load_data(FID_path)

    IS_result = [model, np.average(IS), np.std(IS)]
    FID_result = [model, np.average(FID), np.std(FID)]

    return IS, FID, IS_epoch_data, IS_result, FID_result

# Function to create boxplots for IS and FID scores
def create_boxplots(IS_data, FID_data, model_name_IS, model_name_FID, image_path):
    # Preparing data for boxplots
    box_IS = [[IS.flatten(), name, np.average(IS)] for IS, name in zip(IS_data, model_name_IS)]
    wgan_data = [data for data in box_IS if data[1] == 'WGAN']
    lsgan_data = [data for data in box_IS if data[1] == 'LSGAN']
    other_data = [data for data in box_IS if data[1] not in ['WGAN', 'LSGAN']]

    other_data.sort(key=lambda x: x[2], reverse=True)

    box_IS = wgan_data + lsgan_data + other_data

    box_FID = [[FID.flatten(), name, -100 if name == 'WGAN' else (-1 if name == 'LSGAN' else np.average(FID))] for
               FID, name in zip(FID_data, model_name_FID)]
    box_FID.sort(key=lambda x: x[2])

    # Common color map for models
    unique_models = list(set(model_name_IS + model_name_FID))
    color_map = {model: plt.cm.tab20(i / len(unique_models)) for i, model in enumerate(unique_models)}

    # Creating and saving boxplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 14))
    plot_boxplot(ax1, box_IS, color_map, 'IS Score', linewidth=2.5)
    plot_boxplot_with_twinx(ax2, box_FID, color_map, 'FID Score', linewidth=2.5)
    plt.tight_layout()
    plt.savefig(os.path.join(image_path, 'IS_FID_boxplots.tiff'), dpi=300)
    # plt.show()
    plt.close()

def plot_boxplot(axis, box_data, color_map, ylabel, linewidth=2.5, median_linewidth=2):
    props = dict(linewidth=linewidth)
    median_props = dict(linewidth=median_linewidth, color='red')
    axis.boxplot(list(zip(*box_data))[0], vert=True, patch_artist=True,
                 boxprops=props, whiskerprops=props, capprops=props, medianprops=median_props)
    for patch, model in zip(axis.artists, list(zip(*box_data))[1]):
        patch.set_facecolor(color_map[model])
    for i in range(len(box_data)):
        y = box_data[i][0]
        x = np.random.normal(i + 1, 0.04, size=len(y))
        axis.plot(x, y, 'r.', alpha=0.6, markersize=7)
    axis.set_xticks(range(1, len(box_data) + 1))
    axis.set_xticklabels(list(zip(*box_data))[1], rotation=40, fontsize=25)
    axis.set_ylabel(ylabel, fontsize=25)
    axis.tick_params(axis='y', labelsize=18)
    axis.grid(True, which='both', axis='both')

def plot_boxplot_with_twinx(axis, box_data, color_map, ylabel, linewidth=2.5, median_linewidth=2):
    ax_twin = axis.twinx()
    props = dict(linewidth=linewidth)
    median_props = dict(linewidth=median_linewidth, color='red')
    ax_twin.boxplot(list(zip(*box_data))[0], vert=True, patch_artist=True,
                    boxprops=props, whiskerprops=props, capprops=props, medianprops=median_props)
    for patch, model in zip(ax_twin.artists, list(zip(*box_data))[1]):
        patch.set_facecolor(color_map[model])
    for i in range(len(box_data)):
        y = box_data[i][0]
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax_twin.plot(x, y, 'r.', alpha=0.6, markersize=7)
    axis.set_xticks(range(1, len(box_data) + 1))
    axis.set_xticklabels(list(zip(*box_data))[1], rotation=40, fontsize=25)
    ax_twin.set_ylabel(ylabel, fontsize=25)
    ax_twin.tick_params(axis='y', labelsize=18)
    axis.grid(True, which='both', axis='x')
    ax_twin.grid(True, which='both', axis='y')
    axis.yaxis.set_visible(False)

def load_responses(epoch):
    response = np.load(f"{responses_path}/array/{epoch}.npz")
    return response['generated'], response['real']

# Main processing loop
path = '../../result/'

generated_responses = {}
real_responses = {}
D_epochs = [500, 20000, 40000, 60000, 80000]
final_D_epoch = 100000
bins = 20

for parameter in os.listdir(path):
    parameter_path = os.path.join(path, parameter)
    if os.path.isdir(parameter_path):
        for image in os.listdir(parameter_path):
            if image in ['celeba_images', 'cifar100_images', 'cifar10_images', 'lsun_images']:
                image_path = os.path.join(parameter_path, image)
                if os.path.isdir(image_path):
                    IS_data, FID_data, IS_epoch, IS_result, FID_result = [], [], [], [], []
                    for model in os.listdir(image_path):
                        model_path = os.path.join(image_path, model)
                        if os.path.isdir(model_path):
                            IS, FID, IS_epoch_data, IS_res, FID_res = process_model_data(model_path)
                            IS_data.append(IS)
                            FID_data.append(FID)
                            IS_result.append(IS_res)
                            FID_result.append(FID_res)
                            IS_epoch.append([model, IS_epoch_data])

                            responses_path = os.path.join(model_path, 'responses')
                            if os.path.exists(f"{responses_path}/array/") and os.path.isdir(f"{responses_path}/array/"):
                                # individual_density_plots
                                if len(D_epochs) != 5:
                                    fig, axes = plt.subplots(1, len(D_epochs) + 1, figsize=((len(D_epochs) + 1) * 5, 5))
                                    axes = np.array([axes])
                                else:
                                    fig, axes = plt.subplots(2, 3, figsize=(3 * 5, 2 * 5))
                                    axes = axes.flatten()
                                for i, D_epoch in enumerate(D_epochs + [final_D_epoch]):
                                    alpha_value = 0.8 if D_epoch == final_D_epoch else 0.2 + (i / (len(D_epochs) + 1) * 0.5)
                                    generated, real = load_responses(D_epoch)
                                    ax = axes[i]
                                    sns.kdeplot(generated.ravel(), color='blue', fill=True, alpha=alpha_value,
                                                ax=axes[i])
                                    sns.kdeplot(real.ravel(), color='red', fill=True, alpha=alpha_value,
                                                ax=axes[i])
                                    axes[i].set_title(f'Iteration {D_epoch}')
                                    axes[i].set_xlabel('Discriminator Responses', fontsize=15)
                                    axes[i].set_ylabel('Density', fontsize=15)
                                    axes[i].tick_params(axis='x', labelsize=15)
                                    axes[i].tick_params(axis='y', labelsize=15)
                                    axes[i].legend(['Generated', 'Real'])

                                plt.tight_layout()
                                plt.savefig(f"{responses_path}/individual_density_plots.tiff", dpi=300)
                                # plt.show()
                                plt.clf()

                    min_epoch = 0
                    max_epoch = len(IS_epoch[1][1])-1
                    tick_interval = (max_epoch - min_epoch) / 5
                    tick_positions = np.arange(min_epoch, max_epoch + tick_interval, tick_interval)
                    tick_labels = np.arange(0, 100001, 20000)

                    models_IS = ['WGAN', 'LSGAN', 'Euclidean', 'Chi2']

                    # smoothed graph
                    plt.figure(figsize=(10, 5))
                    plt.grid(True, linestyle='--', alpha=0.5)
                    window_size = 4
                    kernel = np.ones(window_size) / window_size

                    for model_data in IS_epoch:
                        if model_data[0] in models_IS:
                            data_padded = np.pad(model_data[1].reshape(-1), (window_size // 2, window_size // 2 - 1),
                                                 'edge')
                            smoothed_data = np.convolve(data_padded, kernel, 'valid')
                            plt.plot(smoothed_data, label=model_data[0], linewidth=2, linestyle='-')

                    plt.legend(fontsize=10, loc='lower right')
                    plt.xlabel('Iteration', fontsize=16)
                    plt.ylabel('IS Score', fontsize=16)
                    plt.xticks(fontsize=15)
                    plt.xticks(tick_positions, tick_labels)
                    plt.yticks(fontsize=15)
                    plt.tight_layout()
                    plt.savefig(os.path.join(image_path, 'IS_epoch_smoothed.tiff'), dpi=300)
                    plt.clf()
                    plt.close()

                    # Save overall IS, FID score result
                    pd.DataFrame(IS_result).to_csv(os.path.join(image_path, 'IS_all.csv'), header=['model', 'average', 'std'], index=False)
                    pd.DataFrame(FID_result).to_csv(os.path.join(image_path, 'FID_all.csv'), header=['model', 'average', 'std'], index=False)

                    # Read data for IS and FID scores
                    data_IS = pd.read_csv(os.path.join(image_path, "IS_all.csv"))
                    data_FID = pd.read_csv(os.path.join(image_path, "FID_all.csv"))

                    # Extract model names
                    model_name_IS = [name.replace('_', ' ') for name in data_IS['model']]
                    model_name_FID = [name.replace('_', ' ') for name in data_FID['model']]

                    # Create and save boxplots
                    create_boxplots(IS_data, FID_data, model_name_IS, model_name_FID, image_path)

