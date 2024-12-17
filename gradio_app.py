import gradio as gr

import os
import torch
from PIL import Image

from pipeline_freescale import StableDiffusionXLPipeline
from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d

def infer_gpu_part(pipe, seed, prompt, negative_prompt, ddim_steps, guidance_scale, resolutions_list, fast_mode, cosine_scale, disable_freeu):
    pipe = pipe.to("cuda") 
    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(seed)
    if not disable_freeu:
        register_free_upblock2d(pipe, b1=1.1, b2=1.2, s1=0.6, s2=0.4)
        register_free_crossattn_upblock2d(pipe, b1=1.1, b2=1.2, s1=0.6, s2=0.4)
    result = pipe(prompt, negative_prompt=negative_prompt, generator=generator,
                num_inference_steps=ddim_steps, guidance_scale=guidance_scale,
                resolutions_list=resolutions_list, fast_mode=fast_mode, cosine_scale=cosine_scale,
                ).images[0]
    return result

def infer(prompt, output_size, ddim_steps, guidance_scale, cosine_scale, seed, options, negative_prompt):

    disable_freeu = 'Disable FreeU' in options
    fast_mode = 'Fast Mode' in options
    if output_size == "2048 x 2048":
        resolutions_list = [[1024, 1024],
                            [2048, 2048]]
    elif output_size == "1024 x 2048":
        resolutions_list = [[512, 1024],
                            [1024, 2048]]
    elif output_size == "2048 x 1024":
        resolutions_list = [[1024, 512],
                            [2048, 1024]]
    elif output_size == "4096 x 4096":
        resolutions_list = [[1024, 1024],
                            [2048, 2048],
                            [4096, 4096]]
    elif output_size == "2048 x 4096":
        resolutions_list = [[512, 1024],
                            [1024, 2048],
                            [2048, 4096]]
    elif output_size == "4096 x 2048":
        resolutions_list = [[1024, 512],
                            [2048, 1024],
                            [4096, 2048]]

    model_ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionXLPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16)

    print('GPU starts')
    result = infer_gpu_part(pipe, seed, prompt, negative_prompt, ddim_steps, guidance_scale, resolutions_list, fast_mode, cosine_scale, disable_freeu)
    print('GPU ends')

    save_path = 'output.png'
    result.save(save_path)

    return save_path


examples = [
    ["A Enchanted illustration of a Palatial Ghost Explosion with a Mystical Sky, in the style of Eric, viewed from CamProX, Bokeh. High resolution, 8k, insanely detailed.",],
    ["Brunette pilot girl in a snowstorm, full body, moody lighting, intricate details, depth of field, outdoors, Fujifilm XT3, RAW, 8K UHD, film grain, Unreal Engine 5, ray tracing.",],
    ["A cute and adorable fluffy puppy wearing a witch hat in a Halloween autumn evening forest, falling autumn leaves, brown acorns on the ground, Halloween pumpkins spiderwebs, bats, and a witch’s broom.",],
    ["A Fantasy Realism illustration of a Heroic Phoenix Rising Adventurous with a Fantasy Waterfall, in the style of Illusia, viewed from Capture360XPro, Historical light. High resolution, 8k, insanely detailed.",],
]

css = """
#col-container {max-width: 768px; margin-left: auto; margin-right: auto;}
a {text-decoration-line: underline; font-weight: 600;}
.animate-spin {
  animation: spin 1s linear infinite;
}
@keyframes spin {
  from {
      transform: rotate(0deg);
  }
  to {
      transform: rotate(360deg);
  }
}
#share-btn-container {
  display: flex; 
  padding-left: 0.5rem !important; 
  padding-right: 0.5rem !important; 
  background-color: #000000; 
  justify-content: center; 
  align-items: center; 
  border-radius: 9999px !important; 
  max-width: 15rem;
  height: 36px;
}
div#share-btn-container > div {
    flex-direction: row;
    background: black;
    align-items: center;
}
#share-btn-container:hover {
  background-color: #060606;
}
#share-btn {
  all: initial; 
  color: #ffffff;
  font-weight: 600; 
  cursor:pointer; 
  font-family: 'IBM Plex Sans', sans-serif; 
  margin-left: 0.5rem !important; 
  padding-top: 0.5rem !important; 
  padding-bottom: 0.5rem !important;
  right:0;
}
#share-btn * {
  all: unset;
}
#share-btn-container div:nth-child(-n+2){
  width: auto !important;
  min-height: 0px !important;
}
#share-btn-container .wrap {
  display: none !important;
}
#share-btn-container.hidden {
  display: none!important;
}
img[src*='#center'] { 
    display: inline-block;
    margin: unset;
}
.footer {
        margin-bottom: 45px;
        margin-top: 10px;
        text-align: center;
        border-bottom: 1px solid #e5e5e5;
    }
    .footer>p {
        font-size: .8rem;
        display: inline-block;
        padding: 0 10px;
        transform: translateY(10px);
        background: white;
    }
    .dark .footer {
        border-color: #303030;
    }
    .dark .footer>p {
        background: #0b0f19;
    }
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            """
            <h1 style="text-align: center;">FreeScale (unleash the resolution of SDXL)</h1>
            <p style="text-align: center;">
            FreeScale: Unleashing the Resolution of Diffusion Models via Tuning-Free Scale Fusion
            </p>
            <p style="text-align: center;">
            <a href="https://arxiv.org/abs/2412.09626" target="_blank"><b>[arXiv]</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
            <a href="http://haonanqiu.com/projects/FreeScale.html" target="_blank"><b>[Project Page]</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
            <a href="https://github.com/ali-vilab/FreeScale" target="_blank"><b>[Code]</b></a>
            </p>         
            """
        )

        prompt_in = gr.Textbox(label="Prompt", placeholder="A panda walking and munching bamboo in a bamboo forest.")

        with gr.Row():
            with gr.Accordion('FreeScale Parameters (feel free to adjust these parameters based on your prompt): ', open=False):
                with gr.Row():
                    output_size = gr.Dropdown(["2048 x 2048", "1024 x 2048", "2048 x 1024", "1024 x 2048", "4096 x 4096", "2048 x 4096", "4096 x 2048"], value="2048 x 2048", label="Output Size (H x W)", info="Due to GPU constraints, run the demo locally for higher resolutions.", scale=2)
                    options = gr.CheckboxGroup(['Disable FreeU', 'Fast Mode'], label='Options (NOT recommended to change)', scale=1)
                with gr.Row():
                    ddim_steps = gr.Slider(label='DDIM Steps',
                             minimum=5,
                             maximum=200,
                             step=1,
                             value=50)
                    guidance_scale = gr.Slider(label='Guidance Scale',
                             minimum=1.0,
                             maximum=20.0,
                             step=0.1,
                             value=7.5)
                with gr.Row():
                    cosine_scale = gr.Slider(label='Cosine Scale',
                             minimum=0,
                             maximum=10,
                             step=0.1,
                             value=2.0)
                    seed = gr.Slider(label='Random Seed',
                             minimum=0,
                             maximum=10000,
                             step=1,
                             value=123)
                with gr.Row():
                    negative_prompt = gr.Textbox(label='Negative Prompt', value='blurry, ugly, duplicate, poorly drawn, deformed, mosaic')

        submit_btn = gr.Button("Generate", variant='primary')
        image_result = gr.Image(label="Image Output")

        gr.Examples(examples=examples, inputs=[prompt_in, output_size, ddim_steps, guidance_scale, cosine_scale, seed, options, negative_prompt])

    submit_btn.click(fn=infer,
            inputs=[prompt_in, output_size, ddim_steps, guidance_scale, cosine_scale, seed, options, negative_prompt],
            outputs=[image_result],
            api_name="freescalehf")

if __name__ == "__main__":
    demo.queue(max_size=8).launch()