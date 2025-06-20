import math, torch
from functools import partial
from torch import nn, Tensor
from torchvision.transforms.functional import normalize
from transformers import AutoModel
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from .configuration_live import LiveConfigMixin

def _siglip_vision_encode(vision_model: nn.Module, frames: Tensor, frame_token_cls: bool, frame_token_pooled: tuple,
    mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], rescale_factor=0.00392156862745098, **kwargs):
    # move frames to same device as model
    device = next(vision_model.parameters()).device
    frames = frames.to(device)
    frames = normalize(frames * rescale_factor, mean=mean, std=std)
    with torch.cuda.amp.autocast(enabled=False):
        vision_outputs = vision_model(frames)
        last_hidden_state = vision_outputs.last_hidden_state
        if frame_token_pooled:
            s = int(math.sqrt(last_hidden_state.shape[1]))
            spatial_tokens = torch.nn.functional.adaptive_avg_pool2d(
                last_hidden_state.reshape(
                    last_hidden_state.shape[0], s, s, last_hidden_state.shape[-1]
                ).permute(0, 3, 1, 2),
                frame_token_pooled
            ).flatten(2, 3).permute(0, 2, 1)
            if not frame_token_cls:
                return spatial_tokens
        if frame_token_cls:
            cls_token = vision_outputs.pooler_output[:, None]
            if not frame_token_pooled:
                return cls_token
    return torch.cat([cls_token, spatial_tokens], dim=1)

def _clip_vision_encode(vision_model: nn.Module, frames: Tensor, frame_token_cls: bool, frame_token_pooled: tuple,
    mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, rescale_factor=0.00392156862745098, **kwargs):
    # move frames to same device as model
    device = next(vision_model.parameters()).device
    frames = frames.to(device)
    frames = normalize(frames * rescale_factor, mean=mean, std=std)
    with torch.cuda.amp.autocast(enabled=False):
        vision_outputs = vision_model(frames)
        last_hidden_state = vision_outputs.last_hidden_state
        if frame_token_pooled:
            s = int(math.sqrt(last_hidden_state.shape[1]))
            spatial_tokens = torch.nn.functional.adaptive_avg_pool2d(
                last_hidden_state[:,1:].reshape(
                    last_hidden_state.shape[0], s, s, last_hidden_state.shape[-1]
                ).permute(0, 3, 1, 2),
                frame_token_pooled
            ).flatten(2, 3).permute(0, 2, 1)
            if not frame_token_cls:
                return spatial_tokens
        if frame_token_cls:
            cls_token = last_hidden_state[:,0]
            if not frame_token_pooled:
                return cls_token
    return torch.cat([cls_token, spatial_tokens], dim=1)

def build_live_vision(config: LiveConfigMixin):
    model = AutoModel.from_pretrained(config.vision_pretrained).vision_model
    # ensure the vision model is on the same device as input (e.g., GPU) to avoid dtype/device mismatch
    if torch.cuda.is_available():
        model = model.cuda()
    if 'google/siglip-large-patch16-384' == config.vision_pretrained:
        return model, partial(_siglip_vision_encode, frame_token_cls=config.frame_token_cls, frame_token_pooled=tuple(config.frame_token_pooled))
    elif 'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90k' == config.vision_pretrained or 'openai/clip-vit-large-patch14-336' == config.vision_pretrained:
        return model, partial(_clip_vision_encode, frame_token_cls=config.frame_token_cls, frame_token_pooled=tuple(config.frame_token_pooled))
    else:
        raise ValueError(f'Unverified vision_pretrained: {config.vision_pretrained}')