import itertools
import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch import nn
from pathlib import Path


def load_image(img_path, resize=None):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.0


def get_heatmap_path():
    cwd = Path(os.getcwd())
    return cwd / "imgs"/"heatmaps"


def get_source_path():
    cwd = Path(os.getcwd())
    return cwd  /"imgs"/"dataset" 


def clear_heatmaps():
    path = get_heatmap_path()
    # Delete old content
    shutil.rmtree(path)
    create_path(path)


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Path '{path}' created.")


def get_images(labels: list):
    dir_path = get_source_path()
    res = []
    for label in labels:
        subdir_path = dir_path / label
        for file_path in os.listdir(subdir_path):
            image_path = subdir_path / file_path
            if os.path.isfile(image_path):
                res.append((image_path, label))
    return res


def show_attention_map(heatmap, image_path: str,save_path,keyword,filename):
    _, axes = plt.subplots(1, 3, figsize=(15, 8))
    axes[0].matshow(heatmap.squeeze())
    img = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    result_img = (heatmap * 0.4 + img).astype(np.float32)
    axes[1].imshow(img[..., ::-1])
    axes[2].imshow((result_img / 255)[..., ::-1])

    plt.savefig(save_path + f"GradCAM_{filename}_{keyword}.png")
    #plt.show()
    
    return result_img  # this is the line I added 


def create_grid(image_path: str, label: str, result):
    _, axes = plt.subplots(
        len(result), len(list(result.values())[0]) * 2, figsize=(40, 20)
    )
    rows = result.keys()
    cols = list(
        itertools.chain.from_iterable(
            [["", str(type(entry[0]).__name__)] for entry in list(result.values())[0]]
        )
    )
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)
    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, rotation=0, labelpad=30)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    img = cv2.imread(str(image_path))
    for i, (_,heatmaps) in enumerate(result.items()):
        for j,  heatmap in enumerate(heatmaps):
            axes[i][j * 2].matshow(heatmap.squeeze())
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            result_img = (heatmap * 0.4 + img).astype(np.float32)
            axes[i][j * 2 + 1].imshow((result_img / 255)[..., ::-1])

    dest_folder = get_heatmap_path()/label
    create_path(dest_folder)
    dest_path = dest_folder / image_path.name
    plt.savefig(dest_path)


def get_cnn_modules(module, cnn_module_list=[]):
    for child in module.children():
        if type(child) is nn.Conv2d:
            cnn_module_list.append(child)
        elif child.children() is not None:
            cnn_module_list = get_cnn_modules(child, cnn_module_list)
    return cnn_module_list


def get_all_layers(module, layer_list=[]):
    for child in module.children():
        if len(list(child.children())) == 0:
            layer_list.append(child)
        else:
            layer_list = get_all_layers(child, layer_list)
    return layer_list
