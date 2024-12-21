import gradio as gr

import torch

from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d
from pipeline_freescale import StableDiffusionXLPipeline
from pipeline_freescale_turbo import StableDiffusionXLPipeline_Turbo

dtype = torch.float16
device = "cuda"
model_ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
model_ckpt_turbo = "stabilityai/sdxl-turbo"
pipe = StableDiffusionXLPipeline.from_pretrained(model_ckpt, torch_dtype=dtype).to(device)
pipe_turbo = StableDiffusionXLPipeline_Turbo.from_pretrained(model_ckpt_turbo, torch_dtype=dtype).to(device)
register_free_upblock2d(pipe, b1=1.1, b2=1.2, s1=0.6, s2=0.4)
register_free_crossattn_upblock2d(pipe, b1=1.1, b2=1.2, s1=0.6, s2=0.4)
register_free_upblock2d(pipe_turbo, b1=1.1, b2=1.2, s1=0.6, s2=0.4)
register_free_crossattn_upblock2d(pipe_turbo, b1=1.1, b2=1.2, s1=0.6, s2=0.4)
torch.cuda.empty_cache()

def infer_gpu_part(seed, prompt, negative_prompt, ddim_steps, guidance_scale, resolutions_list, fast_mode, cosine_scale, restart_steps):
    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(seed)

    result = pipe(prompt, negative_prompt=negative_prompt, generator=generator,
                num_inference_steps=ddim_steps, guidance_scale=guidance_scale,
                resolutions_list=resolutions_list, fast_mode=fast_mode, cosine_scale=cosine_scale,
                restart_steps=restart_steps,
                ).images[0]
    return result

def infer_gpu_part_turbo(seed, prompt, negative_prompt, ddim_steps, guidance_scale, resolutions_list, fast_mode, cosine_scale, restart_steps):
    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(seed)

    result = pipe_turbo(prompt, negative_prompt=negative_prompt, generator=generator,
                num_inference_steps=ddim_steps, guidance_scale=guidance_scale,
                resolutions_list=resolutions_list, fast_mode=fast_mode, cosine_scale=cosine_scale,
                restart_steps=restart_steps,
                ).images[0]
    return result

def infer(prompt, output_size, ddim_steps, guidance_scale, cosine_scale, seed, options, negative_prompt):

    print(prompt)
    print(negative_prompt)

    disable_turbo = 'Disable Turbo' in options

    if disable_turbo:
        fast_mode = True
        if output_size == "2048 x 2048":
            resolutions_list = [[1024, 1024],
                                [2048, 2048]]
        elif output_size == "1024 x 2048":
            resolutions_list = [[512, 1024],
                                [1024, 2048]]
        elif output_size == "2048 x 1024":
            resolutions_list = [[1024, 512],
                                [2048, 1024]]
        restart_steps = [int(ddim_steps * 0.3)]

        result = infer_gpu_part(seed, prompt, negative_prompt, ddim_steps, guidance_scale, resolutions_list, fast_mode, cosine_scale, restart_steps)

    else:
        fast_mode = False
        if output_size == "2048 x 2048":
            resolutions_list = [[512, 512],
                                [1024, 1024],
                                [2048, 2048]]
        elif output_size == "1024 x 2048":
            resolutions_list = [[256, 512],
                                [512, 1024],
                                [1024, 2048]]
        elif output_size == "2048 x 1024":
            resolutions_list = [[512, 256],
                                [1024, 512],
                                [2048, 1024]]
        restart_steps = [int(ddim_steps * 0.5)] * 2

        result = infer_gpu_part_turbo(seed, prompt, negative_prompt, ddim_steps, guidance_scale, resolutions_list, fast_mode, cosine_scale, restart_steps)

    return result


examples = [
    ["A cute and adorable fluffy puppy wearing a witch hat in a Halloween autumn evening forest, falling autumn leaves, brown acorns on the ground, Halloween pumpkins spiderwebs, bats, and a witch’s broom.",],
    ["Brunette pilot girl in a snowstorm, full body, moody lighting, intricate details, depth of field, outdoors, Fujifilm XT3, RAW, 8K UHD, film grain, Unreal Engine 5, ray tracing.",],
    ["A panda walking and munching bamboo in a bamboo forest.",],
]

css = """
#col-container {max-width: 768px; margin-left: auto; margin-right: auto;}
"""

def mode_update(options):
    if 'Disable Turbo' in options:
        return [gr.Slider(minimum=5,
                        maximum=60,
                        value=50),
                gr.Slider(minimum=1.0,
                        maximum=20.0,
                        value=7.5),
                gr.Row(visible=True)]
    else:
        return [gr.Slider(minimum=2,
                        maximum=6,
                        value=4),
                gr.Slider(minimum=0.0,
                        maximum=1.0,
                        value=0.0),
                gr.Row(visible=False)]

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
            with gr.Accordion('Advanced Settings', open=False):
                with gr.Row():
                    output_size = gr.Dropdown(["2048 x 2048", "1024 x 2048", "2048 x 1024"], value="2048 x 2048", label="Output Size (H x W)", info="Due to GPU constraints, run the demo locally for higher resolutions.")
                    options = gr.CheckboxGroup(['Disable Turbo'], label="Options", info="Disable Turbo will get better results but cost more time.")
                with gr.Row():
                    ddim_steps = gr.Slider(label='DDIM Steps',
                             minimum=2,
                             maximum=6,
                             step=1,
                             value=4)
                    guidance_scale = gr.Slider(label='Guidance Scale (Disabled in Turbo)',
                             minimum=0.0,
                             maximum=1.0,
                             step=0.1,
                             value=0.0)
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
                             value=111)
                with gr.Row() as row_neg:
                    negative_prompt = gr.Textbox(label='Negative Prompt', value='blurry, ugly, duplicate, poorly drawn, deformed, mosaic', visible=False)

        options.change(mode_update, options, [ddim_steps, guidance_scale, row_neg])

        submit_btn = gr.Button("Generate", variant='primary')
        image_result = gr.Image(label="Image Output")

        gr.Examples(examples=examples, inputs=[prompt_in, output_size, ddim_steps, guidance_scale, cosine_scale, seed, options, negative_prompt])

    submit_btn.click(fn=infer,
            inputs=[prompt_in, output_size, ddim_steps, guidance_scale, cosine_scale, seed, options, negative_prompt],
            outputs=[image_result],
            api_name="freescalehf")

if __name__ == "__main__":
    demo.queue(max_size=8).launch()