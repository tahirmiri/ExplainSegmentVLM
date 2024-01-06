import os
import numpy as np
import spacy
import scispacy
nlp = spacy.load('en_core_web_sm')
biomednlp = spacy.load("en_core_sci_lg")
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import random
import matplotlib.colors as colors
import matplotlib.cm as cm
import textwrap


def image_query_extractor(filename,data_path,preprocess_val,device):

     # preparing the img-query couple
     jpg = data_path + f"/{filename}.jpg"
     txt = data_path + f"/{filename}.txt"

     raw_image  = Image.open(jpg) 
     with open(txt, 'r') as text_file: 
          query = str(text_file.read()) 

     filename_dir = f'ExplainSegmentVLM/saliencies/{filename}/'

     if not os.path.exists(filename_dir):
          os.mkdir(filename_dir)                          

     new_file_dir_sliding = f'ExplainSegmentVLM/saliencies/{filename}/sliding_window'          
     os.mkdir(new_file_dir_sliding)
     
     new_file_dir_fixed = f'ExplainSegmentVLM/saliencies/{filename}/fixed_cropsize'
     os.mkdir(new_file_dir_fixed)     

     new_file_dir_mincrop = f'ExplainSegmentVLM/saliencies/{filename}/min_cropsize'
     os.mkdir(new_file_dir_mincrop)     

     new_file_dir_GradCAM = f'ExplainSegmentVLM/saliencies/{filename}/GradCAM'
     os.mkdir(new_file_dir_GradCAM)                    

     #width, height = raw_image.size
     #print(f'The resolution of {filename} is {width} x {height} pixels')

     #image processing part
     img_np_tensor = np.array(raw_image)   
     img_pt_tensor = preprocess_val(raw_image)
     img_pt_tensor = img_pt_tensor.to(device=device, non_blocking=True) #img_pt_tensor = img_pt_tensor.to(device=device, dtype=input_dtype, non_blocking=True)
     img_pt_tensor = img_pt_tensor.unsqueeze(0).to(device=device)

     return raw_image, img_np_tensor,img_pt_tensor, query, filename_dir

def keywords_extractor(query,keyword,tokenizer,device):

     if keyword == "all":
          
          selected_keywords = []
          biomeddoc = biomednlp(query)
          biomed_terms = [ent.text for ent in biomeddoc.ents]
          doc = nlp(query)
          temp = list(doc.noun_chunks)
          noun_chunks = [chunk.text for chunk in temp]
          articles = ["a", "an", "the"]
          for i in range(len(noun_chunks)):
               chunk_words = noun_chunks[i].split()
               if chunk_words[0].lower() in articles:
                    noun_chunks[i] = " ".join(chunk_words[1:])

          standalone_nouns = [token.text for token in doc if token.pos_ in ('NOUN')]
          selected_keywords = biomed_terms + standalone_nouns
          selected_keywords = list(set(selected_keywords)) # no repeated keywords

          all_tokens = [tokenizer(keyword).to(device=device) for keyword in selected_keywords]
          keywords_tokens = dict(zip(selected_keywords, all_tokens))
          print("The list of selected keywords is:\n\n", selected_keywords, flush=True)

     elif keyword != "all":

          selected_keywords = [keyword]
          all_tokens = [tokenizer(keyword).to(device=device) for keyword in selected_keywords]
          keywords_tokens = dict(zip(selected_keywords, all_tokens))
          # print("The list of keywords to be processed is:\n\n", selected_keywords, flush=True)



     return selected_keywords, keywords_tokens

def final_plot(raw_image,sal_gradcam,result_img,sal_fixed,sal_sliding,sal_min,img_np_tensor,query,keyword,filename_dir,filename,crop_size,stride_size,num_crops,vcent):

     # Create a 5x2 subplot
     fig, axs = plt.subplots(5, 2, figsize=(10, 15))

     axs[0, 0].imshow(raw_image)
     axs[0, 0].axis('off')
     axs[0, 0].set_title("Original Image")
    
     axs[1, 0].matshow(sal_gradcam.squeeze())
     axs[1, 0].axis('off')
     axs[1, 0].set_title("GradCAM heatmap")
     axs[1, 1].imshow((result_img / 255)[..., ::-1])
     axs[1, 1].axis('off')
    
     axs[2, 0].imshow(sal_fixed, norm=colors.TwoSlopeNorm(vcenter=vcent), cmap='jet')
     axs[2, 0].axis('off')
     axs[2, 0].set_title("Fixed_size Cropping heatmap")
     axs[2, 1].imshow(img_np_tensor)
     axs[2, 1].imshow(sal_fixed,norm=colors.TwoSlopeNorm(vcenter=vcent),cmap="jet", alpha=0.5)
     axs[2, 1].axis('off')
    
     axs[3, 0].imshow(sal_sliding, norm=colors.TwoSlopeNorm(vcenter=vcent), cmap='jet')
     axs[3, 0].axis('off')
     axs[3, 0].set_title("Sliding Window Cropping heatmap")
     axs[3, 1].imshow(img_np_tensor)
     axs[3, 1].imshow(sal_sliding,norm=colors.TwoSlopeNorm(vcenter=vcent),cmap="jet", alpha=0.5)
     axs[3, 1].axis('off')
    
     axs[4, 0].imshow(sal_min, norm=colors.TwoSlopeNorm(vcenter=vcent), cmap='jet')
     axs[4, 0].axis('off')
     axs[4, 0].set_title("Minimum sized Cropping heatmap")
     axs[4, 1].imshow(img_np_tensor)
     axs[4, 1].imshow(sal_min,norm=colors.TwoSlopeNorm(vcenter=vcent),cmap="jet", alpha=0.5)
     axs[4, 1].axis('off')
     
    
     # Replace axes[0,1] with texts
     #QUERY = "Your long query here... This is a very long query that needs to be split into several lines to ensure it fits within the allocated space."
     #KEYWORD = "Your keyword here..."
     QUERY = query
     KEYWORD = keyword


     # Split QUERY into words
     words = QUERY.split()
     querywords_ranks = {word:0 for word in words}
     for word in words:
          temp = []
          for key in KEYWORD.split():
               if word in key.split():
                    temp.append(1)
          if len(temp) == 0:
               querywords_ranks[word] = 0
          else:
               querywords_ranks[word] = max(temp)

     # Normalize the ranks
     def standardization(querywords_ranks):
          values = np.array(list(querywords_ranks.values()))
          mean = np.mean(values)
          std_dev = np.std(values)
          normalized_values = {word: (value - mean) / std_dev for word, value in querywords_ranks.items()}
          return normalized_values

     normalized_values = standardization(querywords_ranks)

     # Color each word based on its rank
     cmap = cm.get_cmap("coolwarm")
     colored_words = []
     for word in words:
          color = cmap(normalized_values[word])
          colored_words.append(f'{word}')

     # Join all colored words into a single string
     colored_query = ' '.join(colored_words)

     # Split QUERY into lines if it's too long to fit in one line
     max_words = 7 # Adjust this number according to your requirements
     lines = []
     words = colored_query.split()
     while words:
          chunk = ' '.join(words[:max_words])
          words = words[max_words:]
          lines.append(chunk)

     # Add a blank line after QUERY and then add KEYWORD
     spttl = f"Keyword: {keyword}   Patch size: {crop_size}    Stride: {stride_size}    #Patches:{num_crops}"

     lines.append('\n')
     lines.append(KEYWORD)
     lines.append('\n')
     lines.append(spttl)


     #axs[0, 1].text(0.5, 0.5, '\n'.join(lines), va='center', ha='center', color=cmap(normalized_values[word]))
     axs[0, 1].text(0.5, 0.5, '\n'.join(lines), va='center', ha='center')

     # Remove axis for axes[0,1]
     axs[0, 1].axis('off')

     plt.tight_layout()
     plt.savefig(filename_dir + f'{filename}_allsaliencies.jpg', bbox_inches='tight', dpi=250)  
     plt.close(fig)

def saliency_plot_single(raw_image,saliency_map,img_np_tensor,query,keyword_ranks,keyword,filename, filename_dir, method_version,vcent=0,crop_size=None,num_crops=None,stride_size=None):

     fig, axes = plt.subplots(1,3, figsize=(8,4))

     axes[0].imshow(raw_image)
     axes[0].axis("off")

     axes[1].imshow(saliency_map, norm=colors.TwoSlopeNorm(vcenter=vcent), cmap='jet')
     #axes[1].imshow(saliency_map, norm=colors.TwoSlopeNorm(vcenter=saliency_map.mean()), cmap='jet')
     #axes[1].imshow(saliency_map, norm=colors.LogNorm(vmin=saliency_map.min(),vmax=saliency_map.max()), cmap='jet')
     #axes[1].imshow(saliency_map, norm=colors.Normalize(vmin=saliency_map.min(),vmax=saliency_map.max()), cmap='jet')
     axes[1].axis("off")

     axes[2].imshow(img_np_tensor)
     axes[2].imshow(saliency_map,norm=colors.TwoSlopeNorm(vcenter=vcent),cmap="jet", alpha=0.5)
     #axes[2].imshow(saliency_map, norm=colors.TwoSlopeNorm(vcenter=saliency_map.mean()),cmap="jet", alpha=0.5)
     #axes[2].imshow(saliency_map, norm=colors.LogNorm(vmin=saliency_map.min(),vmax=saliency_map.max()),cmap="jet", alpha=0.5)
     #axes[2].imshow(saliency_map, norm=colors.Normalize(vmin=saliency_map.min()),cmap="jet", alpha=0.5)
     axes[2].axis("off")


     words = query.split()
     querywords_ranks = {word:0 for word in words}
     for word in words:
          temp = []
          for key in keyword_ranks.keys():
               if word in key.split():
                    if key == query:
                         continue
                    else:
                         temp.append(keyword_ranks[key])
          if len(temp) == 0:
               querywords_ranks[word] = 0
          else:
               querywords_ranks[word] = max(temp)


     # choose either standardization or normalize_mathematical
     def standardization(querywords_ranks):
          values = np.array(list(querywords_ranks.values()))
          mean = np.mean(values)
          std_dev = np.std(values)
          normalized_values = {word: (value - mean) / std_dev for word, value in querywords_ranks.items()}
          return normalized_values
     
     normalized_values = standardization(querywords_ranks)

     # def normalize_mathematical(querywords_ranks):
     #      values = np.array(list(querywords_ranks.values()))
     #      norm = np.linalg.norm(values)
     #      normalized_values = {word: value / norm for word, value in querywords_ranks.items()}
     #      return normalized_values
     
     # normalized_values = normalize_mathematical(querywords_ranks)

     # max_value = max(counter_dict.values())
     # min_value = min(counter_dict.values())
     #normalized_values = {word: (value - min_value) / (max_value - min_value) for word, value in counter_dict.items()}
     
     cmap = plt.get_cmap("coolwarm")
     colored_title = []
     colored_colors = []
     for word in words:
          color = cmap(normalized_values[word])
          colored_title.append(word)
          colored_colors.append(color)

     def get_title_width(title, fontsize):
          renderer = plt.gcf().canvas.get_renderer()
          text_widths = [renderer.get_text_width_height_descent(word, plt.gca().title._fontproperties, ismath=False)[0] for word in title.split()]
          return sum(text_widths)

     subplot_width = axes[-1].get_window_extent().x1 - axes[0].get_window_extent().x0
     title_width = get_title_width(query, 12)
     title_x = (subplot_width - title_width) / (2 * fig.bbox.width) + axes[0].get_window_extent().x0 / fig.bbox.width

     #title_x = 0.25 - (get_title_width(query, 12) / (2 * fig.bbox.width))
     title_y = 1.05
     for word, color in zip(colored_title, colored_colors):
          t = fig.text(title_x, title_y, word + " ", color=color, fontsize=12)
          title_x += t.get_window_extent(fig.canvas.get_renderer()).width / fig.bbox.width


     # if keyword == query:
     #      fig.suptitle("full caption")
     #      fig.tight_layout()
     #      plt.savefig(f'/p/project/medvl/users/tahir/bestmodel/saliency_maps/Alex_tests/{filename}/fixed_cropsize/{filename}_fullcaption_patches{n_iters}_fixedcropsize{fixed_crop_size}.jpg', bbox_inches='tight', dpi=250) 
     #      per_image_endtime = time.time()
     #      per_image_time = per_image_endtime - unit_start_time
     #      print(f" {filename}_fullcaption processing time: {per_image_time} seconds", flush=True)
     # else:
          
     # annotations = [f"version ={method_version}", f"crop_size = {crop_size}", f"stride = {stride_size}", f"#patches = {num_crops}"]

     # for i, comment in enumerate(annotations):
     #      axes[2].annotate(comment, xy=(1, i*0.1), xycoords='figure fraction', va='bottom', fontsize=8)

     spttl = f"Keyword: {keyword}   Version: {method_version}    Patch size: {crop_size}    Stride: {stride_size}    #Patches:{num_crops}"

     fig.suptitle(spttl)
     fig.tight_layout()
     plt.savefig(filename_dir + f'{method_version}/{filename}_{keyword}.jpg', bbox_inches='tight', dpi=250)  
     # per_image_endtime = time.time()
     # per_image_time = per_image_endtime - unit_start_time
     # print(f" {filename}_{keyword} processing time: {per_image_time} seconds", flush=True)

     plt.close(fig)

