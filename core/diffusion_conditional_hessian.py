import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline  #, EulerDiscreteScheduler

# pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
# pipeline = pipeline.to("cuda")
# # pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
# pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
# pipeline.scheduler.set_timesteps(num_inference_steps=50)
#%%
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    revision="fp16",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()
#%%
# pipe = pipeline
pipe.text_encoder.requires_grad_(False)
pipe.unet.requires_grad_(False)
pipe.vae.requires_grad_(False)
# pipeline.to(torch.half)
#%%
# with torch.autocast("cuda"):
out = pipe("a cute and classy mice wearing dress and heels", )
out.images[0].show()
#%%
out.images[0].save("cute_classy_mice.png")
#%%
@torch.no_grad()
def generate_simplified(
    prompt = ["a lovely cat"],
    negative_prompt = [""],
    num_inference_steps = 50,
    guidance_scale = 7.5):
    # do_classifier_free_guidance
    batch_size = 1
    height, width = 512, 512
    generator = None
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.

    # get prompt text embeddings
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    text_embeddings = pipe.text_encoder(text_input_ids.to(pipe.device))[0]
    bs_embed, seq_len, _ = text_embeddings.shape

    # get negative prompts  text embedding
    max_length = text_input_ids.shape[-1]
    uncond_input = pipe.tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(pipe.device))[0]

    # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
    seq_len = uncond_embeddings.shape[1]
    uncond_embeddings = uncond_embeddings.repeat(batch_size, 1, 1)
    uncond_embeddings = uncond_embeddings.view(batch_size, seq_len, -1)

    # For classifier free guidance, we need to do two forward passes.
    # Here we concatenate the unconditional and text embeddings into a single batch
    # to avoid doing two forward passes
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # get the initial random noise unless the user supplied it
    # Unlike in other pipelines, latents need to be generated in the target device
    # for 1-to-1 results reproducibility with the CompVis implementation.
    # However this currently doesn't work in `mps`.
    latents_shape = (batch_size, pipe.unet.in_channels, height // 8, width // 8)
    latents_dtype = text_embeddings.dtype
    latents = torch.randn(latents_shape, generator=generator, device=pipe.device, dtype=latents_dtype)

    # set timesteps
    pipe.scheduler.set_timesteps(num_inference_steps)
    # Some schedulers like PNDM have timesteps as arrays
    # It's more optimized to move all timesteps to correct device beforehand
    timesteps_tensor = pipe.scheduler.timesteps.to(pipe.device)
    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * pipe.scheduler.init_noise_sigma

    # Main diffusion process
    for i, t in enumerate(pipe.progress_bar(timesteps_tensor)):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        # predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        # compute the previous noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(noise_pred, t, latents, ).prev_sample

    latents = 1 / 0.18215 * latents
    image = pipe.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return image

#%%
image = generate_simplified(
    prompt = ["a lovely cat"],
    negative_prompt = ["Sunshine"],)
# plt_show_image(image[0])

#%%
prompt = ["a cute and classy mice wearing dress and heels"]
text_inputs = pipe.tokenizer(
    prompt,
    padding="max_length",
    max_length=pipe.tokenizer.model_max_length,
    return_tensors="pt",
)
text_input_ids = text_inputs.input_ids
with torch.no_grad():
    text_embeddings = pipe.text_encoder(text_input_ids.to(pipe.device))[0]

#%%
t = 0.5
latent_model_input = torch.randn(1, 4, 64, 64, device=pipe.device, dtype=text_embeddings.dtype)
input_dim = latent_model_input.numel()
#%%
with torch.no_grad():
    noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
#%% Single objective grad
emb_grad = torch.autograd.grad(noise_pred.sum(), text_embeddings_req, retain_graph=True, )[0]
#%% multi objective grad
emb_grad_batch = torch.autograd.grad(noise_pred.flatten(), text_embeddings_req,
                    grad_outputs=torch.eye(input_dim, device="cuda", dtype=torch.half)[:4, :],
                    retain_graph=True, is_grads_batched=True, )[0]
torch.cuda.empty_cache()
#%%
#%%
EPS = 1e-3
Hess = []
for i in tqdm(range(768)):
    perturb_emb = torch.eye(768, device=pipe.device, dtype=text_embeddings.dtype)[:, i].requires_grad_(True)
    text_embeddings_req = text_embeddings.detach().clone()
    text_embeddings_req[0, 3, :] += perturb_emb * EPS
    noise_pred_pert = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings_req).sample
    D2_pert = ((noise_pred_pert - noise_pred.detach())**2).sum()
    D2_pert.backward()
    torch.cuda.empty_cache()
    Hess.append(perturb_emb.grad.detach().clone().cpu())

#%% 5 mins for 768 x 768 Hessian . 6GB memory peak. nor bad.
Hess = torch.stack(Hess, dim=1)
#%%
U, eig = torch.linalg.eig(Hess.float()+Hess.float().T)
#%%
plt.semilogy(U.real)
plt.show()

#%%
emb_grad_batch = torch.autograd.grad(noise_pred.flatten(), text_embeddings_req,
                    grad_outputs=torch.eye(input_dim, device="cuda", dtype=torch.half)[:4, :],
                    retain_graph=True, is_grads_batched=True, )[0]
torch.cuda.empty_cache()

#%%  Hessian of a vector in conditional vectors

perturb_emb = torch.zeros(768, device=pipe.device, dtype=text_embeddings.dtype).requires_grad_(True)
text_embeddings_req = text_embeddings.detach().clone()
text_embeddings_req[0, 3, :] += perturb_emb
#%%
noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings_req).sample
D2 = ((noise_pred - noise_pred.detach())**2).sum()
#%%
grad_0 = torch.autograd.grad(D2, perturb_emb, retain_graph=True, create_graph=True)[0]
torch.cuda.empty_cache()
#%%
#%%
hess_mat = []
for i in tqdm(range(len(grad_0))):
    hess_1 = torch.autograd.grad(grad_0[i], perturb_emb, retain_graph=True, )[0]
    hess_mat.append(hess_1.detach().clone().cpu())
    torch.cuda.empty_cache()
hess_mat = torch.stack(hess_mat, dim=1)
#%
eigval, eigvec = torch.linalg.eig(hess_mat.float() + hess_mat.float().T)
#%%
plt.semilogy(eigval)
plt.show()
