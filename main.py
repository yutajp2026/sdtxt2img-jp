from diffusers import StableDiffusionPipeline
import torch
from translate import Translator
import os

model_file = 'v1-5-pruned-emaonly.safetensors'

if not os.path.exists(model_file):
    url = 'https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors'
    torch.hub.download_url_to_file(url, model_file, hash_prefix=None, progress=True)

device = 'cpu'
pipe = StableDiffusionPipeline.from_single_file(model_file).to(device)
honyaku = Translator('en','ja').translate

while True:
    prompt = input("プロンプトを入力してください: ")
    img = pipe(honyaku(prompt), num_inference_steps=10).images[0]
    img.save('output.jpg')
