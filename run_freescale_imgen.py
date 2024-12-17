import os
import torch
from PIL import Image

from pipeline_freescale_imgen import StableDiffusionXLPipeline
from utils import load_prompts
from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d

img_path = 'imgen_intermediates/tmp_img.png'
mask_path = 'imgen_intermediates/tmp_mask.png'
model_ckpt = "../stable-diffusion-xl-base-1.0"
prompts_file = 'prompts/imgen.txt'
prompts = load_prompts(prompts_file)
# prompts = ['Astronaut on Mars During sunset.']
negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"

folder_name = 'release_4k_imgen'
resolutions_list = [[1024, 1024],
                    [2048, 2048],
                    [4096, 4096]]
cosine_scale = 3.0
cosine_scale_bg = 0.5
disable_freeu = 0
fast_mode = False

pipe = StableDiffusionXLPipeline.from_pretrained(model_ckpt, local_files_only= True, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
if not disable_freeu:
    register_free_upblock2d(pipe, b1=1.1, b2=1.2, s1=0.6, s2=0.4)
    register_free_crossattn_upblock2d(pipe, b1=1.1, b2=1.2, s1=0.6, s2=0.4)

generator = torch.Generator(device='cuda')
generator = generator.manual_seed(123)

os.makedirs(folder_name, exist_ok=True)

for index, prompt in enumerate(prompts):
    print("prompt {}:".format(index))
    print(prompt)
    resuls = pipe(prompt, negative_prompt=negative_prompt, generator=generator,
                num_inference_steps=50, guidance_scale=7.5,
                resolutions_list=resolutions_list, fast_mode=fast_mode, cosine_scale=cosine_scale,
                cosine_scale_bg=cosine_scale_bg, img_path=img_path, mask_path=mask_path,
                )
    for i, result in enumerate(resuls):
        image = result.images[0]
        image.save("{}/img{}_{}.png".format(folder_name, index, resolutions_list[i][0]))
