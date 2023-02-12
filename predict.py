# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/resolve/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
from subprocess import call
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler

from gradio_canny2image import process

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

def download_model(model_name):
    """
    Download model from huggingface with wget and save to models directory
    """
    if not os.path.exists("models"):
        os.mkdir("models")
    if not os.path.exists(f"models/{model_name}"):
        print(f"Downloading {model_name}...")
        call(["wget", "-O", f"models/{model_name}", model_dl_urls[model_name]])

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = create_model('./models/cldm_v15.yaml').cuda()
        self.model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
        self.ddim_sampler = DDIMSampler(model)

    def predict(
        self,
        input_image: Path = Input(description="Grayscale input image"),
        prompt: str = Input(description="Prompt for the model"),
        # model: str = Input(
        #     description="Type of model to use",
        #     choices=["canny", "depth", "hed", "normal", "mlsd", "openpose", "scribble", "seg"],
        #     default="canny"
        # ),
        num_samples: int = Input(description="Number of samples", default=1),
        image_resolution: int = Input(description="Image resolution", default=512),
        low_threshold: int = Input(description="Canny low threshold", default=100),
        high_threshold: int = Input(description="Canny high threshold", default=200),
        ddim_steps: int = Input(description="Steps", default=20),
        scale: float = Input(description="Guidance Scale", default=9.0),
        seed: int = Input(description="Seed", default=0),
        eta: float = Input(description="eta (DDIM)", default=0.0),
        a_prompt: str = Input(description="Added Prompt", default="best quality, extremely detailed"),
        n_prompt: str = Input(description="Negative Prompt", default="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"),
    ) -> Path:
        """Run a single prediction on the model"""
        
        # check that the model is downloaded
        # if not os.path.exists(f"models/{model}"):
        #     download_model(model)

        inputs = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, scale, seed, eta, low_threshold, high_threshold]
        outputs = process(inputs)
        # outputs from numpy to PIL
        outputs = Image.fromarray(outputs)
        # save the output image
        outputs.save("tmp/output.png")
        return Path("tmp/output.png")