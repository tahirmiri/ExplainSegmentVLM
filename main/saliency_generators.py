import torch
from PIL import Image

from gradcam.gradcam import grad_cam,get_layer,get_layer_modif
from gradcam.heatmap import get_heatmaps
from gradcam.utils import (
    create_grid,
    get_all_layers,
    get_images,
    show_attention_map,
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import scipy.stats as stats


# custom modules
from training.saliency import  get_random_crop_params,get_random_crop_params_fixedcropsize,sliding_window,get_cropped_image,update_saliency_map,cosine_similarity


def GradCAM_saliencies(model,preprocess_val,tokenizer,filename,keyword,filename_dir,data_path,device):

     start_time = time.time()

     savepath= filename_dir + 'GradCAM'
     img_path = data_path + f"/{filename}.jpg"

     sal_GradCAM = grad_cam(model,preprocess_val,tokenizer,img_path,keyword,device)
     img_plusGC = show_attention_map(sal_GradCAM,img_path,savepath,keyword,filename) # separately save the GradCAM full results

     end_time = time.time()
     execution_time = end_time - start_time
     print(f"GradCAM processing time: {execution_time} seconds")

     return sal_GradCAM, img_plusGC

def fixed_cropsize_saliencies(model,preprocess_val,tokenizer,device,keyword,num_crops,crop_size,selected_keywords, img_pt_tensor,keywords_tokens,img_np_tensor,query,raw_image,filename_dir):

     start_time = time.time()

     num_patches = num_crops
     fixed_crop_size = crop_size
     
     x_dim, y_dim = raw_image.size     
     total_coverage = np.zeros((y_dim, x_dim))

     with torch.no_grad(): #,torch.cuda.amp.autocast()

          img_embed = model.encode_image(img_pt_tensor) 
          #txt_embed = model.encode_text(keywords_tokens[keyword]) 
          txt_embed = model.encode_text(tokenizer(query).to(device=device)) 
          ground_truth_cossim = cosine_similarity(txt_embed.cpu().numpy(), img_embed.cpu().numpy()).item()
          saliency_map = np.zeros((y_dim, x_dim))

          # words = query.split()
          # counter_dict  = {word:0 for word in words}
          counter_dict = {key:0 for key in selected_keywords}

          
          for _ in range(num_patches):  
               x, y, crop_size = get_random_crop_params_fixedcropsize(y_dim, x_dim, fixed_crop_size)
               im_crop = get_cropped_image(img_np_tensor, x, y, crop_size)  # 2. Getting a random crop of the image
               im_crop = Image.fromarray(im_crop)

               im_crop = preprocess_val(im_crop)
               im_crop = im_crop.to(device=device, non_blocking=True)
               im_crop = im_crop.unsqueeze(0).to(device=device)

               temp_img_embed = model.encode_image(im_crop)
               #temp_txt_embed = txt_embed
               temp_txt_embed = model.encode_text(keywords_tokens[keyword])


               patch_cossim = cosine_similarity(temp_txt_embed.cpu().numpy(), temp_img_embed.cpu().numpy()).item()
               similarity = patch_cossim - ground_truth_cossim
               update_saliency_map(saliency_map, similarity, x, y, crop_size)  # 5. Updating the region on the saliency map
               
               total_coverage[y : y + crop_size, x : x + crop_size] = 1

               #tokens = [tokenizer(word).to(device=device) for word in words]
               tokens = [tokenizer(key).to(device=device) for key in selected_keywords]
               #tokens = [tokenizer(word).cuda() for word in words]
               embeddings = [model.encode_text(token)  for token in tokens]
               cosine_sims = [cosine_similarity(embedding.cpu().numpy(), temp_img_embed.cpu().numpy()).item() for embedding in embeddings]
               ranks = stats.rankdata(cosine_sims)
               #word_ranks = dict(zip(words, ranks))
               keyword_ranks = dict(zip(selected_keywords, ranks))


               # for word, rank in word_ranks.items():
               #      counter_dict[word] += rank    
               for key, rank in keyword_ranks.items():
                    counter_dict[key] += rank 

               fig, ax = plt.subplots()
               ax.imshow(saliency_map, norm=colors.TwoSlopeNorm(vcenter=0), cmap='jet')
               rect = patches.Rectangle((x, y), crop_size, crop_size, linewidth=2, edgecolor='r', facecolor='none')
               ax.add_patch(rect)
               plt.savefig(filename_dir + '/fixed_cropsize/current_patch.jpg')
               plt.close(fig)

          
          total_coverage_image = np.where(total_coverage == 1, 0, 1)
          coverage_percentage = (np.sum(total_coverage) / (x_dim * y_dim)) * 100
          print(f"coverage percentage: {coverage_percentage} %", flush=True)    
          fig_total_coverage, ax_total_coverage = plt.subplots()
          ax_total_coverage.imshow(total_coverage_image, cmap='gray')
          plt.savefig(filename_dir + '/fixed_cropsize/total_coverage.jpg')
          plt.close(fig_total_coverage)


          end_time = time.time()
          execution_time = end_time - start_time
          print(f"Fixed_cropping execution time: {execution_time} seconds")

          return saliency_map, counter_dict, keyword_ranks 

def sliding_window_saliencies(model,preprocess_val,tokenizer,device,keyword,stride_size,crop_size,selected_keywords, img_pt_tensor,keywords_tokens,img_np_tensor,query,raw_image,filename_dir):

     start_time = time.time()


     sliding_crop_size = crop_size
     
     x_dim, y_dim = raw_image.size     
     total_coverage = np.zeros((y_dim, x_dim))


     with torch.no_grad(): #,torch.cuda.amp.autocast()

          img_embed = model.encode_image(img_pt_tensor) 
          #txt_embed = model.encode_text(keywords_tokens[keyword]) 
          txt_embed = model.encode_text(tokenizer(query).to(device=device)) 
          ground_truth_cossim = cosine_similarity(txt_embed.cpu().numpy(), img_embed.cpu().numpy()).item()
          saliency_map = np.zeros((y_dim, x_dim))

          # words = query.split()
          # counter_dict  = {word:0 for word in words}
          counter_dict = {key:0 for key in selected_keywords}
          #np.random.seed(100)
          
          for x, y, crop_size in sliding_window(y_dim, x_dim, sliding_crop_size,stride=stride_size):
               
               im_crop = get_cropped_image(img_np_tensor, x, y, crop_size)  
               im_crop = Image.fromarray(im_crop)

               im_crop = preprocess_val(im_crop)
               im_crop = im_crop.to(device=device, non_blocking=True)
               im_crop = im_crop.unsqueeze(0).to(device=device)

               temp_img_embed = model.encode_image(im_crop)
               #temp_txt_embed = txt_embed
               temp_txt_embed = model.encode_text(keywords_tokens[keyword])

               patch_cossim = cosine_similarity(temp_txt_embed.cpu().numpy(), temp_img_embed.cpu().numpy()).item()
               similarity = patch_cossim - ground_truth_cossim
               update_saliency_map(saliency_map, similarity, x, y, crop_size)  # 5. Updating the region on the saliency map

               total_coverage[y : y + crop_size, x : x + crop_size] = 1


               #tokens = [tokenizer(word).to(device=device) for word in words]
               tokens = [tokenizer(key).to(device=device) for key in selected_keywords]
               #tokens = [tokenizer(word).cuda() for word in words]
               embeddings = [model.encode_text(token)  for token in tokens]
               cosine_sims = [cosine_similarity(embedding.cpu().numpy(), temp_img_embed.cpu().numpy()).item() for embedding in embeddings]
               ranks = stats.rankdata(cosine_sims)
               #word_ranks = dict(zip(words, ranks))
               keyword_ranks = dict(zip(selected_keywords, ranks))


               # for word, rank in word_ranks.items():
               #      counter_dict[word] += rank    
               for key, rank in keyword_ranks.items():
                    counter_dict[key] += rank 


               fig, ax = plt.subplots()
               ax.imshow(saliency_map, norm=colors.TwoSlopeNorm(vcenter=0), cmap='jet')
               rect = patches.Rectangle((x, y), crop_size, crop_size, linewidth=2, edgecolor='r', facecolor='none')
               ax.add_patch(rect)
               plt.savefig(filename_dir + '/sliding_window/current_patch.jpg')
               plt.close(fig)

          
          total_coverage_image = np.where(total_coverage == 1, 0, 1)
          coverage_percentage = (np.sum(total_coverage) / (x_dim * y_dim)) * 100
          print(f"coverage percentage: {coverage_percentage} %", flush=True)    
          fig_total_coverage, ax_total_coverage = plt.subplots()
          ax_total_coverage.imshow(total_coverage_image, cmap='gray')
          plt.savefig(filename_dir + '/sliding_window/total_coverage.jpg')
          plt.close(fig_total_coverage)


          end_time = time.time()

          execution_time = end_time - start_time
          print(f"Sliding_window method execution time: {execution_time} seconds")


          return saliency_map, counter_dict, keyword_ranks

def min_cropsize_saliencies(model,preprocess_val,tokenizer,device,keyword,num_crops,crop_size,selected_keywords, img_pt_tensor,keywords_tokens,img_np_tensor,query,raw_image,filename_dir):

     start_time = time.time()

     num_patches = num_crops
     min_crop_size = crop_size
     
     x_dim, y_dim = raw_image.size     
     total_coverage = np.zeros((y_dim, x_dim))

     with torch.no_grad(): #,torch.cuda.amp.autocast()

          img_embed = model.encode_image(img_pt_tensor) 
          #txt_embed = model.encode_text(keywords_tokens[keyword]) 
          txt_embed = model.encode_text(tokenizer(query).to(device=device)) 
          ground_truth_cossim = cosine_similarity(txt_embed.cpu().numpy(), img_embed.cpu().numpy()).item()
          saliency_map = np.zeros((y_dim, x_dim))
          
          # words = query.split()
          # counter_dict  = {word:0 for word in words}
          counter_dict = {key:0 for key in selected_keywords}
          
          for _ in range(num_patches): 
               x, y, crop_size = get_random_crop_params(y_dim, x_dim, min_crop_size)
               im_crop = get_cropped_image(img_np_tensor, x, y, crop_size)  # 2. Getting a random crop of the image
               im_crop = Image.fromarray(im_crop)

               im_crop = preprocess_val(im_crop)
               im_crop = im_crop.to(device=device, non_blocking=True)
               im_crop = im_crop.unsqueeze(0).to(device=device)

               temp_img_embed = model.encode_image(im_crop)
               #temp_txt_embed = txt_embed
               temp_txt_embed = model.encode_text(keywords_tokens[keyword])

               patch_cossim = cosine_similarity(temp_txt_embed.cpu().numpy(), temp_img_embed.cpu().numpy()).item()
               similarity = patch_cossim - ground_truth_cossim
               update_saliency_map(saliency_map, similarity, x, y, crop_size)  # 5. Updating the region on the saliency map

               total_coverage[y : y + crop_size, x : x + crop_size] = 1

               #tokens = [tokenizer(word).to(device=device) for word in words]
               tokens = [tokenizer(key).to(device=device) for key in selected_keywords]
               #tokens = [tokenizer(word).cuda() for word in words]
               embeddings = [model.encode_text(token)  for token in tokens]
               cosine_sims = [cosine_similarity(embedding.cpu().numpy(), temp_img_embed.cpu().numpy()).item() for embedding in embeddings]
               ranks = stats.rankdata(cosine_sims)
               #word_ranks = dict(zip(words, ranks))
               keyword_ranks = dict(zip(selected_keywords, ranks))


               # for word, rank in word_ranks.items():
               #      counter_dict[word] += rank    
               for key, rank in keyword_ranks.items():
                    counter_dict[key] += rank 
     

               # words_and_sims = dict(zip(words,cosine_sims))
               # temp = sorted(words_and_sims.items(), key=lambda x: x[1])
               # words_sims_ranks = {key: (value, index) for index, (key, value) in enumerate(temp)}

               # for word in words:

               #      if word in all_keywords:
               #           counter_dict[word] += words_sims_ranks[word][1] 
               #      else:
               #           counter_dict[word] += 0


               fig, ax = plt.subplots()
               ax.imshow(saliency_map, norm=colors.TwoSlopeNorm(vcenter=0), cmap='jet')
               rect = patches.Rectangle((x, y), crop_size, crop_size, linewidth=2, edgecolor='r', facecolor='none')
               ax.add_patch(rect)
               plt.savefig(filename_dir + '/min_cropsize/current_patch.jpg')
               plt.close(fig)

          
          total_coverage_image = np.where(total_coverage == 1, 0, 1)
          coverage_percentage = (np.sum(total_coverage) / (x_dim * y_dim)) * 100
          print(f"coverage percentage: {coverage_percentage} %", flush=True)    
          fig_total_coverage, ax_total_coverage = plt.subplots()
          ax_total_coverage.imshow(total_coverage_image, cmap='gray')
          plt.savefig(filename_dir + '/min_cropsize/total_coverage.jpg')
          plt.close(fig_total_coverage)

          end_time = time.time()

          execution_time = end_time - start_time
          print(f"Minimum_cropsiz method execution time: {execution_time} seconds")


          return saliency_map, counter_dict, keyword_ranks