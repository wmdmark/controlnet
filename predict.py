# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/resolve/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
from subprocess import call
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler
from PIL import Image
import numpy as np
from typing import List

# from gradio_canny2image import process_canny
# from gradio_depth2image import process_depth
# from gradio_hed2image import process_hed
# from gradio_normal2image import process_normal
from gradio_hough2image import process_mlsd
# from gradio_scribble2image import process_scribble
# from gradio_seg2image import process_seg

from utils import get_state_dict_path, download_model, model_dl_urls, annotator_dl_urls

MODEL_TYPE = "mlsd"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = create_model('./models/cldm_v15.yaml').cuda()
        self.model.load_state_dict(load_state_dict(get_state_dict_path(MODEL_TYPE), location='cuda'))
        self.ddim_sampler = DDIMSampler(self.model)

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Prompt for the model"),
        # model: str = Input(
        #     description="Type of model to use",
        #     choices=["canny", "depth", "hed", "normal", "mlsd", "openpose", "scribble", "seg"],
        #     default="canny"
        # ),
        num_samples: str = Input(
            description="Number of samples (higher values may OOM)",
            choices=['1', '4'],
            default='1'
        ),
        image_resolution: str = Input(
            description="Image resolution to be generated",
            choices = ['256', '512', '768'],
            default='512'
        ),
        # low_threshold: int = Input(description="Canny low threshold (only applicable when model type is 'canny')", default=100, ge=1, le=255), # only applicable when model type is 'canny'
        # high_threshold: int = Input(description="Canny high threshold (only applicable when model type is 'canny')", default=200, ge=1, le=255), # only applicable when model type is 'canny'
        ddim_steps: int = Input(description="Steps", default=20),
        scale: float = Input(description="Guidance Scale", default=9.0, ge=0.1, le=30.0),
        seed: int = Input(description="Seed", default=None),
        eta: float = Input(description="eta (DDIM)", default=0.0),
        a_prompt: str = Input(description="Added Prompt", default="best quality, extremely detailed"),
        n_prompt: str = Input(description="Negative Prompt", default="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"),
        detect_resolution: int = Input(description="Resolution for detection (only applicable when model type is 'HED', 'Segmentation', or 'MLSD')", default=512, ge=128, le=1024), # only applicable when model type is 'HED', 'seg', or 'MLSD'
        # bg_threshold: float = Input(description="Background Threshold (only applicable when model type is 'normal')", default=0.0, ge=0.0, le=1.0), # only applicable when model type is 'normal'
        value_threshold: float = Input(description="Value Threshold (only applicable when model type is 'MLSD')", default=0.1, ge=0.01, le=2.0), # only applicable when model type is 'MLSD'
        distance_threshold: float = Input(description="Distance Threshold (only applicable when model type is 'MLSD')", default=0.1, ge=0.01, le=20.0), # only applicable when model type is 'MLSD'
    ) -> List[Path]:
        """Run a single prediction on the model"""
        num_samples = int(num_samples)
        image_resolution = int(image_resolution)
        if not seed:
            seed = np.random.randint(1000000)
        else:
            seed = int(seed)

        # load input_image
        input_image = Image.open(image)
        # convert to numpy
        input_image = np.array(input_image)

        # outputs = process_canny(
        #     input_image,
        #     prompt,
        #     a_prompt,
        #     n_prompt,
        #     num_samples,
        #     image_resolution,
        #     ddim_steps,
        #     scale,
        #     seed,
        #     eta,
        #     low_threshold,
        #     high_threshold,
        #     self.model,
        #     self.ddim_sampler,
        # )
        # outputs = process_depth(
        #     input_image,
        #     prompt,
        #     a_prompt,
        #     n_prompt,
        #     num_samples,
        #     image_resolution,
        #     ddim_steps,
        #     scale,
        #     seed,
        #     eta,
        #     self.model,
        #     self.ddim_sampler,
        # )
        # outputs = process_hed(
        #     input_image,
        #     prompt,
        #     a_prompt,
        #     n_prompt,
        #     num_samples,
        #     image_resolution,
        #     detect_resolution,
        #     ddim_steps,
        #     scale,
        #     seed,
        #     eta,
        #     self.model,
        #     self.ddim_sampler,
        # )
        # outputs = process_normal(
        #     input_image,
        #     prompt,
        #     a_prompt,
        #     n_prompt,
        #     num_samples,
        #     image_resolution,
        #     ddim_steps,
        #     scale,
        #     seed,
        #     eta,
        #     bg_threshold,
        #     self.model,
        #     self.ddim_sampler,
        # )
        outputs = process_mlsd(
            input_image,
            prompt,
            a_prompt,
            n_prompt,
            num_samples,
            image_resolution,
            detect_resolution,
            ddim_steps,
            scale,
            seed,
            eta,
            value_threshold,
            distance_threshold,
            self.model,
            self.ddim_sampler,
        )
        # outputs = process_scribble(
        #     input_image,
        #     prompt,
        #     a_prompt,
        #     n_prompt,
        #     num_samples,
        #     image_resolution,
        #     ddim_steps,
        #     scale,
        #     seed,
        #     eta,
        #     self.model,
        #     self.ddim_sampler,
        # )
        # outputs = process_seg(
        #     input_image,
        #     prompt,
        #     a_prompt,
        #     n_prompt,
        #     num_samples,
        #     image_resolution,
        #     detect_resolution,
        #     ddim_steps,
        #     scale,
        #     seed,
        #     eta,
        #     self.model,
        #     self.ddim_sampler,
        # )
        # outputs from list to PIL
        outputs = [Image.fromarray(output) for output in outputs]
        # save outputs to file
        outputs = [output.save(f"tmp/output_{i}.png") for i, output in enumerate(outputs)]
        # return paths to output files
        return [Path(f"tmp/output_{i}.png") for i in range(len(outputs))]
