import argparse

# add command line arg for model type
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="canny", help="Model type to download")
args = parser.parse_args()

MODEL_TYPE = args.model_type

from utils import model_dl_urls, annotator_dl_urls, download_model

for model_name in annotator_dl_urls.keys():
    download_model(model_name, annotator_dl_urls)

download_model(MODEL_TYPE, model_dl_urls)
