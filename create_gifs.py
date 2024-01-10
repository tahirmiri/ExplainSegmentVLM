from PIL import Image
import os

# List of PNG file paths
image_files = ['image1.png', 'image2.png', 'image3.png']  # Add your file paths here
image_names = ["PMC1266360_4852746.jpg","PMC1435922_4842084.jpg","PMC3166809_651659.jpg"]   
head_fusions = ["mean","max","min"] 
        
for image_name in image_names:    
    path_to_rollout = "medAttentionRollout/"  + image_name[:-4] + "/"
    att_imgs = [path_to_rollout + name for name in sorted(os.listdir(path_to_rollout))]
    print(image_name)
    for head_fusion in head_fusions:
        att_head_imgs = filter(lambda x: head_fusion in x and not ".gif" in x, att_imgs)
        
        # gif images
        images = [Image.open(image) for image in att_head_imgs]

        # Save the first image with a .gif extension and append the rest
        gif_path = path_to_rollout + head_fusion + '.gif'  # Specify the path for the output GIF
        images[0].save(gif_path, save_all=True, append_images=images, optimize=False, duration=200, loop=0)

        del images
# 'duration' is the number of milliseconds between frames; you can adjust it
# 'loop=0' means the GIF will loop indefinitely; change to other numbers for specific loop counts
