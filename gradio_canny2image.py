import cv2
import einops
import gradio as gr
import numpy as np
import torch

from cldm.hack import disable_verbosity
disable_verbosity()

from pytorch_lightning import seed_everything
from annotator.util import resize_image_wh, HWC3
from annotator.canny import apply_canny
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler

def process_canny(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, scale, seed, eta, low_threshold, high_threshold, model, ddim_sampler):

    with torch.no_grad():

        # image_resolution is a string like "512x512"
        # get width and height
        image_resolution_w, image_resolution_h = image_resolution.split("x")
        img = resize_image_wh(HWC3(input_image), int(image_resolution_w), int(image_resolution_h))
        H, W, C = img.shape

        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        seed_everything(seed)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)
        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results
