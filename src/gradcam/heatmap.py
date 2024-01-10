import timm
import torch
from open_clip.timm_model import TimmModel
from open_clip.transformer import VisionTransformer
from torch import nn

from .hook import Hook


# https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/vision_transformers.md
def reshape_transform(model, tensor, grid_size):
    tensor.squeeze()
    result = tensor
    if isinstance(model.visual, VisionTransformer):
        result = tensor[1:, :].reshape(
            tensor.size(1), grid_size[0], grid_size[1], tensor.size(2)
        )
    if isinstance(model.visual, TimmModel):
        if isinstance(
            model.visual.trunk, timm.models.vision_transformer.VisionTransformer
        ):
            result = tensor.reshape(
                tensor.size(0), grid_size[0], grid_size[1], tensor.size(2)
            )

    # Bring the channels to the first dimension, like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def get_gradient(model, hook):
    if isinstance(model.visual, VisionTransformer):
        return reshape_transform(model, hook.gradient.float(), model.visual.grid_size)
    if isinstance(model.visual, TimmModel):
        if isinstance(
            model.visual.trunk, timm.models.vision_transformer.VisionTransformer
        ):
            return reshape_transform(model, hook.gradient.float(), (14, 14))
    return hook.gradient.float()


def get_activation(model, hook):
    if isinstance(model.visual, VisionTransformer):
        return reshape_transform(model, hook.activation.float(), model.visual.grid_size)
    if isinstance(model.visual, TimmModel):
        if isinstance(
            model.visual.trunk, timm.models.vision_transformer.VisionTransformer
        ):
            return reshape_transform(model, hook.activation.float(), (14, 14))
    return hook.activation.float()


# https://arxiv.org/abs/1610.02391
def get_heatmap(
    model: nn.Module, input: torch.Tensor, target: torch.Tensor, layer: nn.Module
) -> torch.Tensor:
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()

    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)

    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)

    with torch.cuda.amp.autocast(), torch.autograd.set_detect_anomaly(
        True
    ), torch.set_grad_enabled(True), Hook(layer) as hook:
        image_features = model.encode_image(input)
        text_features = model.encode_text(target)

        # Stefans approach
        # image_features.backward(text_features)

        # Martins approach
        normalized_image_features = image_features / torch.linalg.norm(
            image_features, dim=-1, keepdim=True
        )
        normalized_text_features = text_features / torch.linalg.norm(
            text_features, dim=-1, keepdim=True
        )
        text_probs = 100.0 * normalized_image_features @ normalized_text_features.T
        text_probs[:, 0].backward()

        grad = get_gradient(model, hook)
        act = get_activation(model, hook)

        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel
        # dimension.
        heatmap = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        heatmap = torch.clamp(heatmap, min=0)
        # Normalize the heatmap
        heatmap /= torch.max(heatmap)

    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])

    return heatmap


def get_heatmaps(
    model: nn.Module, input: torch.Tensor, target: torch.Tensor, layers: [nn.Module]
) -> torch.Tensor:
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()

    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)

    hooks = []
    heatmaps = []

    for layer in layers:
        hooks.append(Hook(layer))

    # do forward pass
    with torch.cuda.amp.autocast(), torch.autograd.set_detect_anomaly(
        True
    ), torch.set_grad_enabled(True):
        image_features = model.encode_image(input)
        text_features = model.encode_text(target)

        # output = model.visual(input)
        # output.backward(text_features)

        normalized_image_features = image_features / torch.linalg.norm(
            image_features, dim=-1, keepdim=True
        )
        normalized_text_features = text_features / torch.linalg.norm(
            text_features, dim=-1, keepdim=True
        )
        text_probs = 100.0 * normalized_image_features @ normalized_text_features.T
        text_probs[:, 0].backward()

    for hook in hooks:
        try:
            grad = get_gradient(model, hook)
            act = get_activation(model, hook)

            # Global average pool gradient across spatial dimension
            # to obtain importance weights.
            alpha = grad.mean(dim=(2, 3), keepdim=True)
            # Weighted combination of activation maps over channel
            # dimension.
            heatmap = torch.sum(act * alpha, dim=1, keepdim=True)
            # We only want neurons with positive influence so we
            # clamp any negative ones.
            heatmap = torch.clamp(heatmap, min=0)
            # Normalize the heatmap
            heatmap /= torch.max(heatmap)

            heatmaps.append(heatmap.squeeze().detach().cpu().numpy())
            hook.clear_hook()

        except Exception as e:
            print(e)
            print("did not work for layer:", layer)

    del hooks
    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])

    return heatmaps
