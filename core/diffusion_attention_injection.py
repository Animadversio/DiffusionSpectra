"""Inject attention map into the model
using an attention module that records attention map / substitute attention map
"""
import os
from os.path import join
import numpy as np
import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline  #, EulerDiscreteScheduler
from collections import defaultdict

exproot = r"/home/binxuwang/insilico_exp/Diffusion_AttnMap/StableDiffusion"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    revision="fp16",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()
pipe.text_encoder.requires_grad_(False)
pipe.unet.requires_grad_(False)
pipe.vae.requires_grad_(False)
#%%
attn_module_dict = [
    ("down00", pipe.unet.down_blocks[0].attentions[0]),
    ("down01", pipe.unet.down_blocks[0].attentions[1]),
    ("down10", pipe.unet.down_blocks[1].attentions[0]),
    ("down11", pipe.unet.down_blocks[1].attentions[1]),
    ("down20", pipe.unet.down_blocks[2].attentions[0]),
    ("down21", pipe.unet.down_blocks[2].attentions[1]),
    ("mid00",  pipe.unet.mid_block.attentions[0]),
    ("up10",   pipe.unet.up_blocks[1].attentions[0]),
    ("up11",   pipe.unet.up_blocks[1].attentions[1]),
    ("up20",   pipe.unet.up_blocks[2].attentions[0]),
    ("up21",   pipe.unet.up_blocks[2].attentions[1]),
    ("up30",   pipe.unet.up_blocks[3].attentions[0]),
    ("up31",   pipe.unet.up_blocks[3].attentions[1]),
]
#%%
from diffusers.models.attention import CrossAttention
class CrossAttention_Injector(CrossAttention):
    """Inject attention map into the model"""
    def __init__(self, query_dim: int, cross_attention_dim=None, heads: int = 8, dim_head: int = 64, dropout: int = 0.0):
        CrossAttention.__init__(self, query_dim, cross_attention_dim=cross_attention_dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.attn_map = None
        self.record = False
        self.substitute = False

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        context = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        value = self.to_v(context)
        value = self.reshape_heads_to_batch_dim(value)

        if (not self.record) and self.substitute and (self.attn_map is not None):
            attention_probs = self.attn_map.to(hidden_states.device)
        else:
            query = self.to_q(hidden_states)
            key = self.to_k(context)
            query = self.reshape_heads_to_batch_dim(query)
            key = self.reshape_heads_to_batch_dim(key)
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
            attention_probs = attention_scores.softmax(dim=-1)
        if self.record:
            self.attn_map = attention_probs.detach().clone().cpu()
        hidden_states = torch.matmul(attention_probs, value)
        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return self.to_out[1](self.to_out[0](hidden_states))


def from_CA(CA_module:CrossAttention):
    """initialize a CrossAttention_Injector from a CrossAttention module"""
    self = CrossAttention_Injector(query_dim=CA_module.to_q.in_features,
               cross_attention_dim=CA_module.to_k.in_features,
               heads=CA_module.heads,
               dim_head=CA_module.to_q.out_features // CA_module.heads, )
    self.to_q = CA_module.to_q
    self.to_k = CA_module.to_k
    self.to_v = CA_module.to_v
    self.to_out = CA_module.to_out
    self.scale = CA_module.scale
    self.heads = CA_module.heads
    self.attn_map = None
    self.record = False
    self.substitute = False
    return self
#%%
for name, module in attn_module_dict:
    CA_module = module.transformer_blocks[0].attn2
    print(name)
    injector = from_CA(CA_module)
    module.transformer_blocks[0].attn2 = injector
#%%
for name, module in attn_module_dict:
    CA_module = module.transformer_blocks[0].attn1
    print(name)
    injector = from_CA(CA_module)
    module.transformer_blocks[0].attn1 = injector
#%%
def set_cross_attn_recording(attn_module_dict, record=True, substiute=False):
    for name, module in attn_module_dict:
        CA_module = module.transformer_blocks[0].attn2
        if isinstance(CA_module, CrossAttention_Injector):
            CA_module.record = record
            CA_module.substiute = substiute
            if record:
                CA_module.attn_map = None


def set_self_attn_recording(attn_module_dict, record=True, substiute=False):
    for name, module in attn_module_dict:
        CA_module = module.transformer_blocks[0].attn1
        if isinstance(CA_module, CrossAttention_Injector):
            CA_module.record = record
            CA_module.substiute = substiute
            if record:
                CA_module.attn_map = None
#%%
#%%
set_cross_attn_recording(attn_module_dict, record=False, substiute=False)
set_self_attn_recording(attn_module_dict,  record=False, substiute=False)
#%%
RNGseed = 82  # 246
RNG = torch.cuda.manual_seed(RNGseed)
# prompt = "a fluffy cat throwing a basketball on a playground"
prompt = "a Cat dancing on a ice"
# prompt = "a fish swim in a bowl"
with torch.no_grad():
    out = pipe(prompt, generator=RNG, )

out.images[0].show()
# text_inputs = pipe.tokenizer(prompt, padding="max_length",
#     max_length=pipe.tokenizer.model_max_length,
#     return_tensors="pt",
# )
# tokens_ids_val = text_inputs.input_ids[0][text_inputs.attention_mask[0].bool()]
# tokens_val = pipe.tokenizer.convert_ids_to_tokens(tokens_ids_val)
#%%
plt.figure(figsize=(8,8))
plt.imshow(out.images[0])
plt.axis("off")
plt.tight_layout()
plt.show()

