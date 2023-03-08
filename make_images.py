#!/usr/bin/env python3
import os
import sys
HOME=os.path.dirname(__file__)
sys.path.append(os.path.join(HOME, 'stable-diffusion-webui'))
sys.path.append(os.path.join(HOME, 'stable-diffusion-webui/repositories/k-diffusion'))
sys.path.append(os.path.join(HOME, 'stable-diffusion-webui/repositories/stable-diffusion-stability-ai'))
sys.path.append(os.path.join(HOME, 'stable-diffusion-webui/extensions-builtin'))
import shutil
import random
import numpy as np
import torch
import safetensors
from omegaconf import OmegaConf
from PIL import Image, ImageFilter, ImageOps
import modules.processing
from modules import prompt_parser, devices
from modules import shared, sd_hijack
from modules.shared import opts
import k_diffusion
from Lora import lora
from glob import glob

with open('prompts', 'r') as f:
    prompts = next(f).strip()
    negative_prompts = next(f).strip()

opt_C = 4
opt_f = 8
width = 512
height = 768
#height = 512
global_count = 0
steps = 20
subseed_strength = 0.3
cfg_scale = 7.0
batch_size=4
n_iter=100

#outputs = glob('output-*')
max_old_id = 0
for old_path in glob('output-*'):
    _, n = old_path.split('-')
    n = int(n)
    if n > max_old_id:
        max_old_id = n

if os.path.exists('output'):
    shutil.move('output', 'output-%d' % (max_old_id+1))

shared.cmd_opts.lora_dir = 'model'
outdir = 'output'
os.makedirs(outdir, exist_ok=True)
shutil.copy('prompts', 'output/prompts')

class CFGDenoiser(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.inner_model = model
        self.init_latent = None
        self.step = 0
        self.image_cfg_scale = None

    def combine_denoised(self, x_out, conds_list, uncond, cond_scale):
        denoised_uncond = x_out[-uncond.shape[0]:]
        denoised = torch.clone(denoised_uncond)

        for i, conds in enumerate(conds_list):
            for cond_index, weight in conds:
                denoised[i] += (x_out[cond_index] - denoised_uncond[i]) * (weight * cond_scale)

        return denoised

    def combine_denoised_for_edit_model(self, x_out, cond_scale):
        out_cond, out_img_cond, out_uncond = x_out.chunk(3)
        denoised = out_uncond + cond_scale * (out_cond - out_img_cond) + self.image_cfg_scale * (out_img_cond - out_uncond)

        return denoised

    def forward(self, x, sigma, uncond, cond, cond_scale, image_cond):

        # at self.image_cfg_scale == 1.0 produced results for edit model are the same as with normal sampling,
        # so is_edit_model is set to False to support AND composition.
        #is_edit_model = shared.sd_model.cond_stage_key == "edit" and self.image_cfg_scale is not None and self.image_cfg_scale != 1.0
        is_edit_model = False

        conds_list, tensor = prompt_parser.reconstruct_multicond_batch(cond, self.step)
        uncond = prompt_parser.reconstruct_cond_batch(uncond, self.step)

        assert not is_edit_model or all([len(conds) == 1 for conds in conds_list]), "AND is not supported for InstructPix2Pix checkpoint (unless using Image CFG scale = 1.0)"

        batch_size = len(conds_list)
        repeats = [len(conds_list[i]) for i in range(batch_size)]

        if not is_edit_model:
            x_in = torch.cat([torch.stack([x[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [x])
            sigma_in = torch.cat([torch.stack([sigma[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [sigma])
            image_cond_in = torch.cat([torch.stack([image_cond[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [image_cond])
        else:
            x_in = torch.cat([torch.stack([x[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [x] + [x])
            sigma_in = torch.cat([torch.stack([sigma[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [sigma] + [sigma])
            image_cond_in = torch.cat([torch.stack([image_cond[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [image_cond] + [torch.zeros_like(self.init_latent)])

        #denoiser_params = CFGDenoiserParams(x_in, image_cond_in, sigma_in, state.sampling_step, state.sampling_steps)
        #cfg_denoiser_callback(denoiser_params)
        #x_in = denoiser_params.x
        #image_cond_in = denoiser_params.image_cond
        #sigma_in = denoiser_params.sigma

        if True:
            x_out = torch.zeros_like(x_in)
            batch_size = batch_size*2 if shared.batch_cond_uncond else batch_size
            for batch_offset in range(0, tensor.shape[0], batch_size):
                a = batch_offset
                b = min(a + batch_size, tensor.shape[0])

                if not is_edit_model:
                    c_crossattn = [tensor[a:b]]
                else:
                    c_crossattn = torch.cat([tensor[a:b]], uncond)

                x_out[a:b] = self.inner_model(x_in[a:b], sigma_in[a:b], cond={"c_crossattn": c_crossattn, "c_concat": [image_cond_in[a:b]]})

            x_out[-uncond.shape[0]:] = self.inner_model(x_in[-uncond.shape[0]:], sigma_in[-uncond.shape[0]:], cond={"c_crossattn": [uncond], "c_concat": [image_cond_in[-uncond.shape[0]:]]})

        #denoised_params = CFGDenoisedParams(x_out, state.sampling_step, state.sampling_steps)
        #cfg_denoised_callback(denoised_params)

        devices.test_for_nans(x_out, "unet")

        #if opts.live_preview_content == "Prompt":
        #    sd_samplers_common.store_latent(x_out[0:uncond.shape[0]])
        #elif opts.live_preview_content == "Negative prompt":
        #    sd_samplers_common.store_latent(x_out[-uncond.shape[0]:])

        if not is_edit_model:
            denoised = self.combine_denoised(x_out, conds_list, uncond, cond_scale)
        else:
            assert False
            denoised = self.combine_denoised_for_edit_model(x_out, cond_scale)

        self.step += 1

        return denoised


if not hasattr(torch.nn, 'Linear_forward_before_lora'):
    torch.nn.Linear_forward_before_lora = torch.nn.Linear.forward

if not hasattr(torch.nn, 'Conv2d_forward_before_lora'):
    torch.nn.Conv2d_forward_before_lora = torch.nn.Conv2d.forward

torch.nn.Linear.forward = lora.lora_Linear_forward
torch.nn.Conv2d.forward = lora.lora_Conv2d_forward

shared.options_templates.update(shared.options_section(('extra_networks', "Extra Networks"), {
    "lora_apply_to_outputs": shared.OptionInfo(False, "Apply Lora to outputs rather than inputs when possible (experimental)"),

}))

model_path = os.path.join(HOME, 'v1-inference.yaml')
state_path = os.path.join(HOME, 'chilloutmix_NiPrunedFp32Fix.safetensors')

from ldm.util import instantiate_from_config
sd_config = OmegaConf.load(model_path)
sd_model = instantiate_from_config(sd_config['model'])

print(sd_model.model.conditioning_key)
state_dict = safetensors.torch.load_file(state_path)
sd_model.load_state_dict(state_dict, strict=False)
del state_dict

sd_hijack.model_hijack.hijack(sd_model)
shared.sd_model = sd_model

lora.list_available_loras()
lora.assign_lora_names_to_compvis_modules(sd_model)
lora.load_loras(['lora'])
#lora.load_loras(['koreanDollLikeness_v10'])

devices.dtype = torch.float16
devices.dtype_vae = torch.float16
devices.dtype_unet = sd_model.model.diffusion_model.dtype
devices.unet_needs_upcast = shared.cmd_opts.upcast_sampling and devices.dtype == torch.float16 and devices.dtype_unet == torch.float16


sd_model.to(shared.device)
sd_model.half()
sd_model.eval()


#sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)


#prompts = 'high quality, looking at viewer, highres, (raw photo:1.2), ((photorealistic:1.4)), best quality ,masterpiece, illustration, an extremely delicate and beautiful, extremely detailed , Amazing, finely detail, masterpiece,best quality,huge filesize, ultra-detailed, highres, extremely detailed,beautiful detailed girl, extremely detailed eyes and face, beautiful detailed eyes,light on face,cinematic lighting, 1girl, smile, full body'

#negative_prompts = 'sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, bad anatomy,(long hair:1.4),DeepNegative,(fat:1.2),facing away, looking away,tilted head, lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark, username,blurry,bad feet,cropped,poorly drawn hands,poorly drawn face,mutation,deformed,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,text,error,missing fingers,missing arms,missing legs,extra digit, extra arms, extra leg, extra foot, multiple girls'


def get_seed():
    return int(random.randrange(4294967294))

prompts = [prompts] * batch_size
negative_prompts = [negative_prompts] * batch_size

uc = prompt_parser.get_learned_conditioning(sd_model, negative_prompts, steps)
c = prompt_parser.get_multicond_learned_conditioning(sd_model, prompts, steps)

def txt2img_image_conditioning(sd_model, x):
    return x.new_zeros(x.shape[0], 5, 1, 1, dtype=x.dtype, device=x.device)

def decode_first_stage(model, x):
    with devices.autocast(disable=x.dtype == devices.dtype_vae):
        x = model.decode_first_stage(x)
    return x

def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm*high_norm).sum(1)

    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)

    omega = torch.acos(dot)
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res

def create_random_tensors(shape, seeds, subseeds=None, subseed_strength=0.0):
    eta_noise_seed_delta = opts.eta_noise_seed_delta or 0
    xs = []
    for i, seed in enumerate(seeds):
        noise_shape = shape 

        subnoise = None
        if subseeds is not None:
            subseed = 0 if i >= len(subseeds) else subseeds[i]

            subnoise = devices.randn(subseed, noise_shape)

        noise = devices.randn(seed, noise_shape)

        if subnoise is not None:
            noise = slerp(subseed_strength, noise, subnoise)
        xs.append(noise)

    x = torch.stack(xs).to(shared.device)
    return x

model_wrap = k_diffusion.external.CompVisDenoiser(sd_model, quantize=shared.opts.enable_quantization)
model_wrap_cfg = CFGDenoiser(model_wrap)

out_images = []
with torch.no_grad(): #, p.sd_model.ema_scope():
    for n in range(n_iter):
        seeds = [get_seed()] * batch_size
        subseeds = [get_seed() for i in range(batch_size)]


        with devices.without_autocast() if devices.unet_needs_upcast else devices.autocast():
            x = create_random_tensors([opt_C, height // opt_f, width // opt_f], seeds=seeds, subseeds=subseeds, subseed_strength=subseed_strength)
            image_cond = txt2img_image_conditioning(sd_model, x)
            extra_args = {
                    'cond': c,
                    'image_cond': image_cond,
                    'uncond': uc,
                    'cond_scale': cfg_scale
                    }

            sigmas = k_diffusion.sampling.get_sigmas_karras(steps, sigma_min=0.1, sigma_max=10, device=shared.device)
            #print(sigmas)
            x *= sigmas[0]
            samples_ddim = k_diffusion.sampling.sample_dpmpp_2m(model_wrap_cfg, x, sigmas, extra_args)

        with devices.autocast(disable=x.dtype == devices.dtype_vae):
            x_samples_ddim = sd_model.decode_first_stage(samples_ddim).cpu().float()
        del samples_ddim

        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples_ddim = x_samples_ddim.numpy()

        devices.torch_gc()

        for i, x_sample in enumerate(x_samples_ddim):
            x_sample = 255. * np.moveaxis(x_sample, 0, 2)
            x_sample = x_sample.astype(np.uint8)

            image = Image.fromarray(x_sample)

            image.save('%s/%03d.png' % (outdir, global_count))
            global_count += 1

        del x_samples_ddim


