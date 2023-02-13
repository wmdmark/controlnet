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

from gradio_canny2image import process_canny
from gradio_depth2image import process_depth
from gradio_hed2image import process_hed
from gradio_normal2image import process_normal
from gradio_hough2image import process_mlsd

model_dl_urls = {
    "canny": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth",
    "depth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_depth.pth",
    "hed": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_hed.pth",
    "normal": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_normal.pth",
    "mlsd": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_mlsd.pth",
    "openpose": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_openpose.pth",
    "scribble": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_scribble.pth",
    "seg": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_seg.pth",
}

annotator_dl_urls = {
    "body_pose_model.pth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/body_pose_model.pth",
    "dpt_hybrid-midas-501f0c75.pt": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt",
    "hand_pose_model.pth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/hand_pose_model.pth",
    "mlsd_large_512_fp32.pth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_large_512_fp32.pth",
    "mlsd_tiny_512_fp32.pth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_tiny_512_fp32.pth",
    "network-bsds500.pth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/network-bsds500.pth",
    "upernet_global_small.pth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/upernet_global_small.pth",
}

def download_model(model_name, urls_map):
    """
    Download model from huggingface with wget and save to models directory
    """
    model_url = urls_map[model_name]
    relative_path_to_model = model_url.replace("https://huggingface.co/lllyasviel/ControlNet/resolve/main/", "")
    if not os.path.exists(relative_path_to_model):
        print(f"Downloading {model_name}...")
        call(["wget", "-O", relative_path_to_model, model_url])

def get_state_dict_path(model_name):
    """
    Get path to model state dict
    """
    return f"./models/control_sd15_{model_name}.pth"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = create_model('./models/cldm_v15.yaml').cuda()
        # self.model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
        self.ddim_sampler = DDIMSampler(self.model)

    def predict(
        self,
        input_image: Path = Input(description="Grayscale input image"),
        prompt: str = Input(description="Prompt for the model"),
        model: str = Input(
            description="Type of model to use",
            choices=["canny", "depth", "hed", "normal", "mlsd", "openpose", "scribble", "seg"],
            default="canny"
        ),
        num_samples: int = Input(description="Number of samples", default=1),
        image_resolution: int = Input(description="Image resolution", default=512),
        low_threshold: int = Input(description="Canny low threshold (only applicable when model type is 'canny')", default=100, ge=1, le=255),
        high_threshold: int = Input(description="Canny high threshold (only applicable when model type is 'canny')", default=200, ge=1, le=255),
        ddim_steps: int = Input(description="Steps", default=20),
        scale: float = Input(description="Guidance Scale", default=9.0),
        seed: int = Input(description="Seed", default=0),
        eta: float = Input(description="eta (DDIM)", default=0.0),
        a_prompt: str = Input(description="Added Prompt", default="best quality, extremely detailed"),
        n_prompt: str = Input(description="Negative Prompt", default="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"),
        detect_resolution: int = Input(description="Resolution for detection (only applicable when model type is 'HED')", default=512, ge=128, le=1024),
        bg_threshold: float = Input(description="Background Threshold (only applicable when model type is 'normal')", default=0.0, ge=0.0, le=1.0),
        value_threshold: float = Input(description="Value Threshold (only applicable when model type is 'MLSD')", default=0.1, ge=0.01, le=2.0),
        distance_threshold: float = Input(description="Distance Threshold (only applicable when model type is 'MLSD')", default=0.1, ge=0.01, le=20.0),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        # load state dict
        print("Loading state dict for model...")
        self.model.load_state_dict(load_state_dict(get_state_dict_path(model), location='cuda'))

        # load input_image
        input_image = Image.open(input_image)
        # convert to numpy
        input_image = np.array(input_image)

        if model == "canny":
            outputs = process_canny(
                input_image,
                prompt,
                a_prompt,
                n_prompt,
                num_samples,
                image_resolution,
                ddim_steps,
                scale,
                seed,
                eta,
                low_threshold,
                high_threshold,
                self.model,
                self.ddim_sampler,
            )
        elif model == "depth":
            outputs = process_depth(
                input_image,
                prompt,
                a_prompt,
                n_prompt,
                num_samples,
                image_resolution,
                ddim_steps,
                scale,
                seed,
                eta,
                self.model,
                self.ddim_sampler,
            )
        elif model == "hed":
            outputs = process_hed(
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
                self.model,
                self.ddim_sampler,
            )
        elif model == "normal":
            outputs = process_normal(
                input_image,
                prompt,
                a_prompt,
                n_prompt,
                num_samples,
                image_resolution,
                ddim_steps,
                scale,
                seed,
                eta,
                bg_threshold,
                self.model,
                self.ddim_sampler,
            )
        elif model == "mlsd":
            outputs = process_mlsd(
                input_image,
                prompt,
                a_prompt,
                n_prompt,
                num_samples,
                image_resolution,
                ddim_steps,
                scale,
                seed,
                eta,
                value_threshold,
                distance_threshold,
                self.model,
                self.ddim_sampler,
            )
        # outputs from list to PIL
        outputs = [Image.fromarray(output) for output in outputs]
        # save outputs to file
        outputs = [output.save(f"tmp/output_{i}.png") for i, output in enumerate(outputs)]
        # return paths to output files
        return [Path(f"tmp/output_{i}.png") for i in range(len(outputs))]