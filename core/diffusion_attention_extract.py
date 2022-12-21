import os
from os.path import join
import numpy as np
import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline  #, EulerDiscreteScheduler

exproot = r"/home/binxuwang/insilico_exp/Diffusion_Hessian/StableDiffusion"
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
#%% Extract the attention mask from the model
from collections import defaultdict

def hook_forger(key, ingraph=False):
    def output_hook(module, input, output):
        activation[key].append(output.detach().cpu())

    return output_hook


activation = defaultdict(list)
def hook_spatial_transformer(module, module_id):
    """Hook the spatial transformer module to extract the QK for computing attention map"""
    h1 = module.transformer_blocks[0].attn1.to_q.register_forward_hook(hook_forger(module_id+"_SelfA_Q"))
    h2 = module.transformer_blocks[0].attn1.to_k.register_forward_hook(hook_forger(module_id+"_SelfA_K"))

    h3 = module.transformer_blocks[0].attn2.to_q.register_forward_hook(hook_forger(module_id+"_CrosA_Q"))
    h4 = module.transformer_blocks[0].attn2.to_k.register_forward_hook(hook_forger(module_id+"_CrosA_K"))
    return [h1, h2, h3, h4]
#%%
from diffusers.models.attention import CrossAttention
#%% Extract all attention modules
attn_module_dict = [
    ("down00", pipe.unet.down_blocks[0].attentions[0]),
    ("down01", pipe.unet.down_blocks[0].attentions[1]),
    ("down10", pipe.unet.down_blocks[1].attentions[0]),
    ("down11", pipe.unet.down_blocks[1].attentions[1]),
    ("down20", pipe.unet.down_blocks[2].attentions[0]),
    ("down21", pipe.unet.down_blocks[2].attentions[1]),
    ("mid00", pipe.unet.mid_block.attentions[0]),
    ("up10", pipe.unet.up_blocks[1].attentions[0]),
    ("up11", pipe.unet.up_blocks[1].attentions[1]),
    ("up20", pipe.unet.up_blocks[2].attentions[0]),
    ("up21", pipe.unet.up_blocks[2].attentions[1]),
    ("up30", pipe.unet.up_blocks[3].attentions[0]),
    ("up31", pipe.unet.up_blocks[3].attentions[1]),
]
#%%
hook_handles = []
for module_id, module in attn_module_dict:
    handles = hook_spatial_transformer(module, module_id)
    hook_handles.extend(handles)
#%%
activation = defaultdict(list)
prompt = "a cute and classy mice wearing dress and heels"
with torch.no_grad():
    out = pipe(prompt, )

out.images[0].show()
# %
for hook in hook_handles:
    hook.remove()
#%%
print([*activation])
#%%
for key in activation:
    activation[key] = torch.stack(activation[key], dim=0)
#%%
module_id = "mid00" #"up10"#"down20"
print(activation[module_id+"_CrosA_Q"].shape)
print(activation[module_id+"_CrosA_K"].shape)
#%%
module_id = "up10"
tstep = 15
Q = activation[module_id+"_CrosA_Q"]
K = activation[module_id+"_CrosA_K"]
scale = K.shape[-1] ** -0.5
# dotmap = torch.baddbmm(Q[tstep].float(), K[tstep].transpose(-1, -2).float(),
#                        beta=0, alpha=scale)
dotmap = scale * torch.bmm(Q[tstep].float(), K[tstep].transpose(-1, -2).float(),)
scoremap = torch.softmax(dotmap, dim=-1)

#%%
plt.figure(figsize=(10, 10))
sns.heatmap(dotmap[0,:,:13].detach().cpu().numpy())
plt.show()
#%%
plt.figure(figsize=(10, 10))
sns.heatmap(dotmap[0, :, 51].reshape(16, 16).detach().cpu().numpy())
plt.axis("image")
plt.show()
#%%
plt.figure(figsize=(10, 10))
plt.imshow(out.images[0])
plt.show()
#%%

#%%
text_inputs = pipe.tokenizer(prompt, padding="max_length",
    max_length=pipe.tokenizer.model_max_length,
    return_tensors="pt",
)
tokens_ids_val = text_inputs.input_ids[0][text_inputs.attention_mask[0].bool()]
tokens_val = pipe.tokenizer.convert_ids_to_tokens(tokens_ids_val)
#%%

#%%
from core.utils.plot_utils import saveallforms
def visualize_dot_maps(dotmap, tokens_val, module_id, tstep, expdir, save=True):
    """

    :param dotmap:  shape [2, S x S, 77]
    :param tokens_val:  shape [T] smaller than 77, the actual tokens without paddings
    :param module_id:  the module id to print on title
    :param tstep:    the time step to print on title
    :param expdir:  the directory to save the figure
    :return:
    """
    for Lyr in range(dotmap.shape[0]):
        space_dim = int(np.sqrt(dotmap.shape[1]))
        fig, axs = plt.subplots(ncols=5, nrows=3, figsize=(11, 6))
        for ax, tok_loc in zip(axs.ravel(), range(len(tokens_val))):
            cim = ax.imshow(dotmap[Lyr, :, tok_loc].reshape(space_dim, space_dim).detach().cpu().numpy(), cmap="rocket")
            fig.colorbar(cim, ax=ax)
            # sns.heatmap(dotmap[Lyr, :, tok_loc].reshape(space_dim, space_dim).detach().cpu().numpy(), ax=ax)
            ax.axis("image")
            ax.set_title(tokens_val[tok_loc])
            ax.set_xticks([])
            ax.set_yticks([])
        plt.suptitle(f"Cross Attention (Dot prod) Map\nModule {module_id} tstep {tstep} Lyr {Lyr}")
        for ax in axs.ravel()[len(tokens_val):]:
            ax.axis("off")
        plt.tight_layout()
        if save:
            saveallforms(expdir, f"{module_id}_CrossAttnMap_t{tstep}_Lyr{Lyr}")
            # plt.savefig(join(expdir, f"{module_id}_CrossAttnMap_t{tstep}.png"))
            plt.close()
        else:
            plt.show()
    return fig
#%%
# Implementation of attention map in the diffusers repo.
# https://github.com/huggingface/diffusers/blob/4125756e88e82370c197fecf28e9f0b4d7eee6c3/src/diffusers/models/cross_attention.py#L214
exproot = r"/home/binxuwang/insilico_exp/Diffusion_AttnMap/StableDiffusion"
expdir = join(exproot, "cute_classy_mice")
os.makedirs(expdir, exist_ok=True)
for module_id in tqdm(["down00", "down01", "down10", "down11", "down20", "down21",
                  "mid00", "up10", "up11", "up20", "up21", "up30", "up31"]):
    for tstep in tqdm(range(0, 51, 5)):
        Q = activation[module_id+"_CrosA_Q"]
        K = activation[module_id+"_CrosA_K"]
        scale = K.shape[-1] ** -0.5
        dotmap = scale * torch.bmm(Q[tstep].float(), K[tstep].transpose(-1, -2).float(),)
        scoremap = torch.softmax(dotmap, dim=-1)
        visualize_dot_maps(dotmap, tokens_val, module_id, tstep, expdir)
        # for Lyr in range(0, 1):
        #     plt.figure(figsize=(10, 10))
        #     sns.heatmap(dotmap[Lyr, :, 51].reshape(16, 16).detach().cpu().numpy())
        #     plt.axis("image")
        #     plt.savefig(join(exproot, f"{module_id}_CrossAttnMap_t{tstep}_L{Lyr}.png"))
        #     plt.close()
#%%
attn_temp = pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1
#%%
from einops import rearrange
module_id =  "up20"
Q = activation[module_id+"_CrosA_Q"]
K = activation[module_id+"_CrosA_K"]
Q_Tmerg = rearrange(Q, "T B I H -> (T B) I H")
Q_head = rearrange(Q_Tmerg, "B I (h H) -> (B h) I H", h=8)
K_Tmerg = rearrange(K, "T B I H -> (T B) I H")
K_head = rearrange(K_Tmerg, "B I (h H) -> (B h) I H", h=8)
dotmaps = scale * torch.bmm(Q_head.float(), K_head.transpose(-1, -2).float(),)
scoremaps = torch.softmax(dotmaps, dim=-1)
dotmap_tsr = rearrange(dotmaps, "(T B h) I K -> T B h I K", B=2, h=8)
scoremap_tsr = rearrange(scoremaps, "(T B h) I K -> T B h I K", B=2, h=8)
# Q_head2 = attn_temp.reshape_heads_to_batch_dim(Q_Tmerg)
# torch.allclose(Q_head, Q_head2, rtol=1E-4)
#%%
figh = visualize_dot_maps(scoremap_tsr[:,:,:,:,:].mean(dim=(0,2)), tokens_val, module_id, tstep, expdir,save=False)
figh.show()
#%%
figh = visualize_dot_maps(dotmap, tokens_val, module_id, tstep, expdir)
figh.show()
#%%
out.images[0].save(join(expdir, "Sample.png"), )
#%%
torch.save(activation, join(expdir, "Attn_QK_activation.pt"))
#%%
dotmap_dict = {}
for module_id in tqdm(["down00", "down01", "down10", "down11", "down20", "down21",
                  "mid00", "up10", "up11", "up20", "up21", "up30", "up31"]):
    for tstep in tqdm(range(0, 51, 5)):
        Q = activation[module_id+"_CrosA_Q"]
        K = activation[module_id+"_CrosA_K"]
        scale = K.shape[-1] ** -0.5
        dotmap = scale * torch.bmm(Q[tstep].float(), K[tstep].transpose(-1, -2).float(),)
    dotmap_dict[module_id+"_CrosA"] = dotmap

#%%
# pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1.to_q
# pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1.to_k
# pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn2.to_q
# pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn2.to_k

pipe.unet.down_blocks[0].attentions[1]

#%%

pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn2
pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1

#%% Dev zone
titlestr = ""
# plt.figure(figsize=(10, 10))
for Lyr in [0, 1]:
    fig, axs = plt.subplots(ncols=5, nrows=3, figsize=(10, 7))
    for ax, tok_loc in zip(axs.ravel(), range(len(tokens_val))):
        sns.heatmap(dotmap[Lyr, :, tok_loc].reshape(16, 16).detach().cpu().numpy(), ax=ax)
        ax.axis("image")
        ax.set_title(tokens_val[tok_loc])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle(f"Cross Attention (Dot prod) Map Lyr {Lyr}\n{titlestr}")
    for ax in axs.ravel()[len(tokens_val):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
# plt.savefig(join(exproot, f"{module_id}_CrossAttnMap_t{tstep}.png"))
# plt.close()
