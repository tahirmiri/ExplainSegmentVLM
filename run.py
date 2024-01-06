import open_clip 
from open_clip import create_model_and_transforms, get_tokenizer
import torch
import pandas as pd
import numpy as np
import os
import argparse
import time

# custom modules 
from main.utils import * 
from main.saliency_generators import *


start_time = time.time()

# 0. processing arguments  ( nargs='*' means interpret input as a list)
'''
Parameters are defined as follows: 
- (filename) if a specific filename is provided, then only that file will be processed, otherwise all image-caption pairs in the data_path will be processed 
- (data_path) the path to the directory containing the image-caption pairs to be processed
- (version) the version of the saliency generation method to be used among (fixed_cropsize, min_cropsize, sliding_window, gradcam, all)
now come hyperparams:
- (crop_size) crop_size x crop_size pixels patches to be processed at a time 
- (stride_size) stride size for sliding window method 
- (num_crops) total number of patches to be processed 
- (keywords) in case a specific keyword is provided, then only that keyword will be processed, otherwise all keywords extracted from the query will be processed

'''


parser = argparse.ArgumentParser(description='Explainability script')
parser.add_argument('--filename', type=str,default="all")
parser.add_argument('--data_path', type=str, default='/p/project/medvl/users/tahir/ExplainSegmentVLM/sample_meddata')
parser.add_argument('--version', type=str, default='all', choices=['fixed_cropsize', 'min_cropsize', 'sliding_window', 'gradcam','all'])
parser.add_argument('--crop_size', type=int, default=64)
parser.add_argument('--stride_size',type=int,  default=64)
parser.add_argument('--num_crops', type=int, default=50)
parser.add_argument('--keyword', type=str, default='all')

args = parser.parse_args()

filename = args.filename
data_path = args.data_path 
version = args.version
crop_size = args.crop_size
#stride_size = args.stride_size if args.version in ['sliding_window','all'] else None
stride_size = args.stride_size
#num_crops = args.num_crops if args.version in ['min_cropsize','fixed_cropsize','all'] else None
num_crops = args.num_crops
keyword = args.keyword



# 1. setup the model, preprocessor, tokenizer, device, and other necessary setups
model, preprocess_train, preprocess_val = create_model_and_transforms("hf-hub_microsoft_pretrained_laion_large", pretrained='/p/project/medvl/models/newest_caption_title_umls_sentence_umls.pt') # /local-scratch/users/thieme/log/mpox_v2/checkpoints/epoch_4.pt
tokenizer = get_tokenizer("hf-hub_microsoft_pretrained_laion_large")
precision = "amp"
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()


# 3.1 process the image and query, and extract the keywords
# 3.2 run all saliency methods for each keyword of a single image at a time, and save the results
# 3.3 after plotting saliency per method individually, add it also to the final 5x2 plot, and save it too 

def generator(filename):
     
     raw_image, img_np_tensor,img_pt_tensor, query, filename_dir = image_query_extractor(filename,data_path,preprocess_val,device)
     selected_keywords, keywords_tokens = keywords_extractor(query,keyword,tokenizer,device)
     
     for selected_keyword in selected_keywords:

          #gradcam
          sal_gradcam,img_plusGC = GradCAM_saliencies(model,preprocess_val,tokenizer,filename,selected_keyword,filename_dir,data_path,device)

          #fixed cropsize
          sal_fixed, counter_dict, keyword_ranks  = fixed_cropsize_saliencies(model,preprocess_val,tokenizer,device,selected_keyword,num_crops,crop_size,selected_keywords, img_pt_tensor,keywords_tokens,img_np_tensor,query,raw_image,filename_dir)
          saliency_plot_single(raw_image,sal_fixed,img_np_tensor,query,keyword_ranks,selected_keyword,filename, filename_dir, method_version='fixed_cropsize',vcent=0,crop_size=crop_size,num_crops=num_crops,stride_size=None)
          
          #sliding window
          sal_sliding, counter_dict, keyword_ranks = sliding_window_saliencies(model,preprocess_val,tokenizer,device,selected_keyword,stride_size,crop_size,selected_keywords, img_pt_tensor,keywords_tokens,img_np_tensor,query,raw_image,filename_dir)
          saliency_plot_single(raw_image,sal_sliding,img_np_tensor,query,keyword_ranks,selected_keyword,filename, filename_dir, method_version='sliding_window',vcent=0,crop_size=crop_size,num_crops=None,stride_size=stride_size)

          #min_cropsize
          sal_min, counter_dict, keyword_ranks  = min_cropsize_saliencies(model,preprocess_val,tokenizer,device,selected_keyword,num_crops,crop_size,selected_keywords, img_pt_tensor,keywords_tokens,img_np_tensor,query,raw_image,filename_dir)
          saliency_plot_single(raw_image,sal_min,img_np_tensor,query,keyword_ranks,selected_keyword,filename, filename_dir, method_version='min_cropsize',vcent=0,crop_size=crop_size,num_crops=num_crops,stride_size=None)
          
          #final 5x2 plot of all saliencies
          final_plot(raw_image,sal_gradcam,img_plusGC,sal_fixed,sal_sliding,sal_min,img_np_tensor,query,selected_keyword,filename_dir,filename,crop_size,stride_size,num_crops,vcent=0)


if filename != "all":
     generator(filename)

     end_time = time.time()
     execution_time = end_time - start_time
     print(f"All saliencies generation time per image-keyword pair: {execution_time} seconds")

elif filename == "all":
     
     for files in os.listdir(data_path):
          filename = files[:-4]
          generator(filename)

          end_time = time.time()
          execution_time = end_time - start_time
          print(f"All saliencies generation time per image-keyword pair: {execution_time} seconds")

else: 
     print("Please enter a valid filename or 'all_files' to process all files in the data_path")



