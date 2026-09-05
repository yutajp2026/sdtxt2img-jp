from diffusers import StableDiffusionPipeline
import torch
from translate import Translator
import os

model_file = 'v1-5-pruned-emaonly.safetensors'

if not os.path.exists(model_file):
    url = 'https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors'
    torch.hub.download_url_to_file(url, model_file, hash_prefix=None, progress=True)

if(torch.cuda.is_available()):
    device = 'cuda'
elif(torch.backends.mps.is_available()):
    device = 'mps'
else:
    device = 'cpu'

pipe = StableDiffusionPipeline.from_single_file(model_file).to(device)
generator = torch.Generator(device)
honyaku = Translator('en','ja').translate

while True:
    prompt = input("Enter your prompt: ")
    img = pipe(honyaku(prompt), width = 256, height = 256, num_inference_steps=10, generator=generator).images[0]
    img.save('output.jpg')
