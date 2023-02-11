# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path

model_dl_urls = {
    "control_sd15_canny.pth": "https://huggingface.co/lllyasviel/ControlNet/blob/main/models/control_sd15_canny.pth",
    "control_sd15_depth.pth": "https://huggingface.co/lllyasviel/ControlNet/blob/main/models/control_sd15_depth.pth",
    "control_sd15_hed.pth": "https://huggingface.co/lllyasviel/ControlNet/blob/main/models/control_sd15_hed.pth",
    "control_sd15_normal.pth": "https://huggingface.co/lllyasviel/ControlNet/blob/main/models/control_sd15_normal.pth",
    "control_sd15_mlsd.pth": "https://huggingface.co/lllyasviel/ControlNet/blob/main/models/control_sd15_mlsd.pth",
    "control_sd15_openpose.pth": "https://huggingface.co/lllyasviel/ControlNet/blob/main/models/control_sd15_openpose.pth",
    "control_sd15_scribble.pth": "https://huggingface.co/lllyasviel/ControlNet/blob/main/models/control_sd15_scribble.pth",
    "control_sd15_seg.pth": "https://huggingface.co/lllyasviel/ControlNet/blob/main/models/control_sd15_seg.pth",
}

annotator_dl_urls = {
    "body_pose_model.pth": "https://huggingface.co/lllyasviel/ControlNet/blob/main/annotator/ckpts/body_pose_model.pth",
    "dpt_hybrid-midas-501f0c75.pt": "https://huggingface.co/lllyasviel/ControlNet/blob/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt",
    "hand_pose_model.pth": "https://huggingface.co/lllyasviel/ControlNet/blob/main/annotator/ckpts/hand_pose_model.pth",
    "mlsd_large_512_fp32.pth": "https://huggingface.co/lllyasviel/ControlNet/blob/main/annotator/ckpts/mlsd_large_512_fp32.pth",
    "mlsd_tiny_512_fp32.pth": "https://huggingface.co/lllyasviel/ControlNet/blob/main/annotator/ckpts/mlsd_tiny_512_fp32.pth",
    "network-bsds500.pth": "https://huggingface.co/lllyasviel/ControlNet/blob/main/annotator/ckpts/network-bsds500.pth",
    "upernet_global_small.pth": "https://huggingface.co/lllyasviel/ControlNet/blob/main/annotator/ckpts/upernet_global_small.pth",
}

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

    def predict(
        self,
        image: Path = Input(description="Grayscale input image"),
        scale: float = Input(
            description="Factor to scale image by", ge=0, le=10, default=1.5
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
