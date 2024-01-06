from pathlib import Path
from typing import Tuple, Union, Iterable

from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import tqdm
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# MIN CROP SIZE CASE
def get_random_crop_params(
    image_height: int, image_width: int, min_crop_size: int
) -> Tuple[int, int, int, int]:
    #np.random.seed(101)
    crop_size = np.random.randint(min_crop_size, min(image_height, image_width))
    x = np.random.randint(image_width - crop_size + 1)
    y = np.random.randint(image_height - crop_size + 1)
    return x, y, crop_size


#FIXED CROP SIZE CASE
def get_random_crop_params_fixedcropsize(
    image_height: int, image_width: int, crop_size: int
) -> Tuple[int, int, int, int]:
    #np.random.seed(101)
    x = np.random.randint(image_width - crop_size + 1)
    y = np.random.randint(image_height - crop_size + 1)
    return x, y, crop_size

#SLIDING WINDOW PRINCIPLE
def sliding_window(image_height: int, image_width: int, crop_size: int, stride=10):
    # stride_size must be divisible by 240, then remove +1 
    for x in range(0, image_width-crop_size+1, stride):
        for y in range(0, image_height-crop_size+1, stride):
            yield x, y, crop_size

#SLIDING WINDOW PRINCIPLE
def sliding_window_largestride(image_height: int, image_width: int, crop_size: int, stride_size=24) -> Iterable[Tuple[int, int, int, int]]:
    stride = stride_size # must be divisible by 240, then remove +1 
    for x in range(0, image_width-crop_size+1, stride):
        for y in range(0, image_height-crop_size+1, stride):
            yield x, y, crop_size


def get_cropped_image(
    im_tensor: np.array, x: int, y: int, crop_size: int
) -> np.array:
    return im_tensor[
        y : y + crop_size,
        x : x + crop_size,
        ...
    ]


def update_saliency_map(
    saliency_map: np.array, similarity: float, x: int, y: int, crop_size: int
) -> None:
    saliency_map[
        y : y + crop_size,
        x : x + crop_size,
    ] += similarity
    return saliency_map

# def update_saliency_map_for_words(saliency_map,similarity) -> None:
#     saliency_map[:,:] += similarity
#     return saliency_map 

def cosine_similarity(
    one: Union[np.ndarray, torch.Tensor], other: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    return one @ other.T / (np.linalg.norm(one) * np.linalg.norm(other))


def plot_saliency_map(image_tensor: np.array, saliency_map: np.array, query: str) -> None:
    plt.figure(dpi=150)
    plt.imshow(image_tensor)
    plt.imshow(
    saliency_map, 
    norm=colors.TwoSlopeNorm(vcenter=0), 
    cmap="jet", 
    alpha=0.5,  # make saliency map trasparent to see original picture
    )
    #plt.title(query,fontsize=6)
    plt.title(f"{query}",fontsize=6)
    plt.axis("off")
    plt.show()
    plt.savefig("original_with_saliency.jpg") 