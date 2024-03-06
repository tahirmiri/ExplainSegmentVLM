'''
initial copy from https://github.com/jacobgil/vit-explain/blob/main/vit_explain.py
'''

import argparse
import sys
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2


from open_clip import create_model_and_transforms, get_tokenizer
from src.rollout.vit_rollout import VITAttentionRollout

# extra patch extraction libraries
from matplotlib import pyplot as plt
from skimage import measure
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--head_fusion', type=str, default='mean',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    return args


## ORIGINAL 

# def show_mask_on_image(img, mask):
#     img = np.float32(img) / 255
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
#     cv2.imwrite("/p/project/medvl/users/tahir/ExplainSegmentVLM/saliencies/PMC1266360_4852746/attention_rollout/heatmap_notnormal.png",heatmap)
#     heatmap = np.float32(heatmap) / 255
#     cv2.imwrite("/p/project/medvl/users/tahir/ExplainSegmentVLM/saliencies/PMC1266360_4852746/attention_rollout/heatmap_normalized.png",heatmap)
#     cam = heatmap + np.float32(img)
#     cam = cam / np.max(cam)
#     cv2.imwrite("/p/project/medvl/users/tahir/ExplainSegmentVLM/saliencies/PMC1266360_4852746/attention_rollout/cam.png",cam)
#     return np.uint8(255 * cam)

## CHECKPOINT 1 (red boundary boxes)

# def show_mask_on_image(img, mask):
#      img = np.float32(img) / 255
#      heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
#      heatmap = np.float32(heatmap) / 255
#      cam = heatmap + np.float32(img)
#      cam = cam / np.max(cam)
#      cam_image = np.uint8(255 * cam)


#      # Convert heatmap to HSV
#      hsv = cv2.cvtColor(cam_image, cv2.COLOR_BGR2HSV)

#      # Define a broader range of red'ish colors in HSV
#      # Adjust these values based on your specific color shades
#      lower_redish = np.array([0, 50, 50])  # lower boundary for hue (include more orange/yellow)
#      upper_redish = np.array([75, 255, 255])  # upper boundary for hue (up to where you still consider it "red'ish")

#      # Threshold the HSV image to get only red'ish colors
#      redish_mask = cv2.inRange(hsv, lower_redish, upper_redish)

#      # Find contours
#      contours, _ = cv2.findContours(redish_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#      # Draw red bounding box for each contour
#      for cnt in contours:
#           x, y, w, h = cv2.boundingRect(cnt)
#           cv2.rectangle(cam_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

#      output_path = "/p/project/medvl/users/tahir/ExplainSegmentVLM/CHECKPOINT#1.png"
#      cv2.imwrite(output_path, cam_image)

#      return cam_image


## CHECKPOINT 2 (red, orange, yellow, black boundary boxes)

# def show_mask_on_image(img, mask):
#     img = np.float32(img) / 255
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255
#     cam = heatmap + np.float32(img)
#     cam = cam / np.max(cam)
#     cam_image = np.uint8(255 * cam)

#     # Convert heatmap to HSV
#     hsv = cv2.cvtColor(cam_image, cv2.COLOR_BGR2HSV)

#     # Define ranges for different colors and their respective boundary box colors
#     color_ranges = {
#           ( tuple([7,  255,  255])): (0,  0,  255),          # Red boundary boxes
#           ( tuple([20,  255,  255])): (0,  165, 255),         # Orange boundary boxes
#           ( tuple([50,  255,  255])): (0,  255,  255),         # Yellow boundary boxes
#           ( tuple([75,  255,  255])): (0,  0,  0),           # Black boundary boxes
#      }


#     # Iterate through the defined color ranges
#     for upper_redish, box_color in color_ranges.items():
#         # Threshold the HSV image to get only specific colors
#         lower_redish = np.array([0, 50, 50])
#         upper_redish = np.array(list(upper_redish))
#         mask = cv2.inRange(hsv, lower_redish, upper_redish)

#         # Find contours
#         contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#         # Draw boundary box for each contour if it's larger than the minimum size
#         for cnt in contours:
#             x, y, w, h = cv2.boundingRect(cnt)
#             if w >= 2 and h >= 2:  # Check if the box is larger than 2x2 pixels
#                 cv2.rectangle(cam_image, (x, y), (x + w, y + h), box_color, 2)

#     output_path = "/p/project/medvl/users/tahir/ExplainSegmentVLM/CHECKPOINT#1.png"
#     cv2.imwrite(output_path, cam_image)

#     return cam_image


## CHECKPOINT 3 ( ) 

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam_image = np.uint8(255 * cam)

    # Convert heatmap to HSV
    hsv = cv2.cvtColor(cam_image, cv2.COLOR_BGR2HSV)

    # Initialize a list to store cluster statistics
    clusters_stats = []

    # Define ranges for different colors and their respective boundary box colors
    color_ranges = {
        (tuple([7, 255, 255])): ('red', (0, 0, 255)), # red
        (tuple([25, 255, 255])): ('orange', (0, 165, 255)), # orange
        (tuple([50, 255, 255])): ('yellow', (0, 255, 255)), # yellow
    }

    # Iterate through the defined color ranges
    cluster_label = 0
    for  upper_hsv, (bb_color, box_color) in color_ranges.items():
        # Threshold the HSV image to get only specific colors
        lower_hsv = np.array([0, 50, 50])
        upper_hsv = np.array(list(upper_hsv))
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw boundary box for each contour if it's larger than the minimum size
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w >= 5 and h >= 5:  # Check if the box is larger than 2x2 pixels
                cv2.rectangle(cam_image, (x, y), (x + w, y + h), box_color, 2)
                
                # Collect statistics
                central_point = (x + w // 2, y + h // 2)
                clusters_stats.append({
                    'label': cluster_label,
                    'length': h,
                    'width': w,
                    'central_point_x': central_point[0],
                    'central_point_y': central_point[1],
                    'bb_color': bb_color
                })
                cluster_label += 1

    # Save the image with colored boundary boxes
    output_path = "/p/project/medvl/users/tahir/ExplainSegmentVLM/CHECKPOINT#1.png"
    cv2.imwrite(output_path, cam_image)

    # Return the image and the statistics
    return cam_image, clusters_stats



if __name__ == '__main__':
    args = get_args()
    image_name = "/p/project/medvl/users/tahir/ExplainSegmentVLM/sample_meddata/PMC1266360_4852746.jpg"
    #    image_name = "/p/project/medvl/users/tahir/ExplainSegmentVLM/sample_meddata/PMC3166809_651659.jpg"
     #image_name = "/p/project/medvl/users/tahir/ExplainSegmentVLM/sample_meddata/PMC1435922_4842084.jpg"

    model, preprocess_train, preprocess_val = create_model_and_transforms("hf-hub_microsoft_pretrained_laion_large", pretrained="/p/project/medvl/models/newest_caption_title_umls_sentence_umls.pt") 
    tokenizer = get_tokenizer("hf-hub_microsoft_pretrained_laion_large")
    
    #print(model)
    #print(type(model))
    #print("model visual type",type(model.visual))
    img = Image.open(image_name)
    input_tensor = preprocess_train(img).unsqueeze(0)

    if args.use_cuda:
        input_tensor = input_tensor.cuda()
      

    print("Doing Attention Rollout")
    attention_rollout = VITAttentionRollout(model)
    mask = attention_rollout(model,input_tensor,head_fusion=args.head_fusion, 
        discard_ratio=args.discard_ratio)
    #cv2.imwrite("/p/project/medvl/users/tahir/ExplainSegmentVLM/saliencies/PMC1266360_4852746/attention_rollout/mask.png",mask)
    name = "/p/project/medvl/users/tahir/ExplainSegmentVLM/saliencies/PMC1266360_4852746/attention_rollout/attention_rollout_{:.3f}_{}.png".format(args.discard_ratio, args.head_fusion)
    
    np_img = np.array(img)
    if len(np_img.shape) == 2:
        #transform black and white into rgb
        np_img = np.stack((np_img,)*3, axis=-1) 
        
    np_img = np_img[:, :, ::-1] #reorder rgb channels
    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    #cv2.imwrite("/p/project/medvl/users/tahir/ExplainSegmentVLM/saliencies/PMC1266360_4852746/attention_rollout/resized_mask.png",mask)
    #mask = show_mask_on_image(np_img, mask)
    #cv2.imwrite(name, mask)
    


    # Assuming you've called show_mask_on_image and received clusters_stats
    mask, clusters_stats = show_mask_on_image(np_img, mask)
    cv2.imwrite(name, mask)

     # Create and save a DataFrame
    df = pd.DataFrame(clusters_stats)
    df.set_index('label', inplace=True)
    df.to_csv("/p/project/medvl/users/tahir/ExplainSegmentVLM/attnrollout_patchstats.csv")
