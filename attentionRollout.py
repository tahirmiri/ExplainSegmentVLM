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

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    cv2.imwrite("/p/project/medvl/users/tahir/ExplainSegmentVLM/saliencies/PMC1266360_4852746/attention_rollout/heatmap_notnormal.png",heatmap)
    heatmap = np.float32(heatmap) / 255
    cv2.imwrite("/p/project/medvl/users/tahir/ExplainSegmentVLM/saliencies/PMC1266360_4852746/attention_rollout/heatmap_normalized.png",heatmap)
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("/p/project/medvl/users/tahir/ExplainSegmentVLM/saliencies/PMC1266360_4852746/attention_rollout/cam.png",cam)
    return np.uint8(255 * cam)

if __name__ == '__main__':
    args = get_args()
    image_name = "/p/project/medvl/users/tahir/ExplainSegmentVLM/sample_meddata/PMC1266360_4852746.jpg"

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
    mask = show_mask_on_image(np_img, mask)
    cv2.imwrite(name, mask)
    