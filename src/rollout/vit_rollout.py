'''
inital copy from https://github.com/jacobgil/vit-explain/blob/main/
'''
import torch
import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2
from open_clip.timm_model import TimmModel

def old_rollout(attentions, discard_ratio, head_fusion):
    #print("attentions length", len(attentions))
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            #print("attention shape",attention.shape)
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=0)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=0)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=0)[0]
            else:
                raise "Attention head fusion type Not supported"
            
            #print("attention_heads_fused shape",attention_heads_fused.shape)
            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0)*attention_heads_fused.size(0))
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    #print("result shape after rollout", result.shape)
    mask = result[0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    ##print(mask)
    return mask      

class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop'):
        
        if isinstance(model.visual,TimmModel):
            #important  for grapping the attn_maps with attn_drop layer
            for module in model.visual.trunk.blocks:
                module.attn.fused_attn = False
                #print("fused_attn",module.attn.fused_attn)
        for name, module in model.visual.trunk.blocks.named_modules():
            if attention_layer_name in name:
                #print(name)
                module.register_forward_hook(self.get_attention)
                
        self.attentions = []

    def get_attention(self, module, input, output):
        #print("layer called")
        self.attentions.append(input[0].cpu().squeeze())
        
    def __call__(self, model, input_tensor,head_fusion="mean",
        discard_ratio=0.9):
        #print("input_tensor shape",input_tensor.shape)
        self.attentions = []
        self.q = []
        self.k = []
        with torch.no_grad():
            output = model.visual(input_tensor)
            
        for q,k in zip(self.q, self.k):
            q = q * model.visual.trunk.blocks[0].attn.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            #print("attn shape after calculation",attn.shape)
            self.attentions.append(attn.cpu().squeeze())
            print(attn)

        return old_rollout(self.attentions, discard_ratio, head_fusion)
