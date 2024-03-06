'''
initial copy from https://github.com/jacobgil/vit-explain/blob/main/vit_explain.py
'''

import argparse
import gc

import sys
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import os
import open_clip

from open_clip import create_model_and_transforms, get_tokenizer

from src.rollout.vit_rollout import VITAttentionRollout

import torch 
if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False
    
# def show_mask_on_image(img, mask):
#     img = np.float32(img) / 255
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255
#     cam = heatmap + np.float32(img)
#     cam = cam / np.max(cam)
#     result = np.uint8(255 * cam)
    
#     del img 
#     del heatmap
#     del cam
#     return result


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

    model, preprocess_train, preprocess_val = create_model_and_transforms("hf-hub_microsoft_pretrained_laion_large", pretrained="/p/project/medvl/models/newest_caption_title_umls_sentence_umls.pt") 
    tokenizer = open_clip.get_tokenizer("hf-hub_microsoft_pretrained_laion_large")
    #model = model.cuda()
    print(model)
    print(type(model))
    

    image_names = ["PMC1266360_4852746.jpg","PMC1435922_4842084.jpg","PMC3166809_651659.jpg"]   
    head_fusions = ["mean","max","min"] 
    discard_ratios =[i*5/100 for i in range(0,20)]
    
    attention_rollout = VITAttentionRollout( model)
        
    for image_name in image_names:    
        img = Image.open("/p/project/medvl/users/tahir/ExplainSegmentVLM/sample_meddata/"+image_name)
        input_tensor = preprocess_train(img).unsqueeze(0)
        if use_cuda:
            input_tensor = input_tensor.cuda()
        saving_path = "/p/project/medvl/users/tahir/ExplainSegmentVLM/saliencies/PMC1266360_4852746/attention_rollout/"  + image_name[:-4] + "/"
        print(image_name)
        os.makedirs(saving_path, exist_ok=True)
        for head_fusion in head_fusions:
            print(head_fusion)
            for discard_ratio in discard_ratios:
                print(discard_ratio)
                mask = attention_rollout(model,input_tensor,head_fusion,discard_ratio)
                name = saving_path + "attention_rollout_{}_{:.3f}.png".format(head_fusion,discard_ratio)
                
                np_img = np.array(img)
                if len(np_img.shape) == 2:
                   #transform black and white into rgb
                   np_img = np.stack((np_img,)*3, axis=-1) 
                mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
                mask = show_mask_on_image(np_img, mask)
                cv2.imwrite(name, mask)
                
                del np_img
                del mask
                
                torch.cuda.empty_cache()
                gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
