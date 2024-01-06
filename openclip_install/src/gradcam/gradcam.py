# https://colab.research.google.com/github/kevinzakka/clip_playground/blob/main/CLIP_GradCAM_Visualization.ipynb

import open_clip
import timm
from open_clip.factory import HF_HUB_PREFIX
from open_clip.modified_resnet import ModifiedResNet
from open_clip.timm_model import TimmModel
from open_clip.transformer import VisionTransformer
from PIL import Image

from .heatmap import get_heatmap
from .utils import show_attention_map


def get_layer(model):
    if isinstance(model.visual, ModifiedResNet):
        return model.visual.layer4
    if isinstance(model.visual, TimmModel):
        if isinstance(
            model.visual.trunk, timm.models.vision_transformer.VisionTransformer
        ):
            return model.visual.trunk.blocks[-1].norm1
        return model.visual.trunk.stages[-1]
    if isinstance(model.visual, VisionTransformer):
        return model.visual.transformer.resblocks[-1].ln_1
    return None

def grad_cam(model, preprocessor,tokenizer,img_path, caption_text,device):
#def grad_cam(model_name, pretrain_tag, image_path, caption_text):
    # if model_name.startswith(HF_HUB_PREFIX):
    #     model, preprocess = open_clip.create_model_from_pretrained(model_name)
    #     tokenizer = open_clip.get_tokenizer(model_name)
    #     print(
    #         f"Executing gradcam for model '{model_name}' with image '{image_path}' and caption '{caption_text}'"
    #     )
    # else:
    #     model, _, preprocess = open_clip.create_model_and_transforms(
    #         model_name, pretrained=pretrain_tag
    #     )
    #     tokenizer = open_clip.get_tokenizer(model_name)
    #     print(
    #         f"Executing gradcam for model '{model_name}' - '{pretrain_tag}' with image '{image_path}' and caption '{caption_text}'"
    #     )

    model = model
    model = model.to(device)
    preprocess = preprocessor
    tokenizer = tokenizer

    image = preprocess(Image.open(img_path)).unsqueeze(0)
    image = image.to(device, non_blocking=True)

    #caption = tokenizer([caption_text])
    caption = tokenizer(caption_text).to(device=device)


    heatmap = get_heatmap(
        model,
        image,
        caption,
        get_layer(model),
    )
    heatmap = heatmap.squeeze().detach().cpu().numpy()
    #show_attention_map(heatmap, image_path)

    return heatmap 

#this is my addition ---------------------------------------------------

def get_last_layer(model):
    return list(model.children())[-1]

def get_layer_modif2(model):
    if isinstance(model.visual, ModifiedResNet):
        return model.visual.layer4
    if isinstance(model.visual, TimmModel):
        return get_last_layer(model.visual)
    if isinstance(model.visual, VisionTransformer):
        return model.visual.transformer.resblocks[-1].ln_1
    return None

def get_layer_modif(model):
    if isinstance(model.visual, ModifiedResNet):
        return model.visual.layer4
    if isinstance(model.visual, TimmModel):
        return model.visual.trunk.blocks[-1].norm1
    if isinstance(model.visual, VisionTransformer):
        return model.visual.transformer.resblocks[-1].ln_1
    return None

# --------------------------------------------------------------------


