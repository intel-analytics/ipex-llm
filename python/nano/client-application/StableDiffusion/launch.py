import gradio as gr
import os
from stable_diffusion.runner import StableDiffusionRunner
import psutil
from launch_full import pipeline, pipeline_process
from stable_diffusion.converter import convert, model_version_map 
from stable_diffusion.pipelines import NanoStableDiffusionPipeline
from stable_diffusion.utils import get_snapshot_dir_from_model_version
from stable_diffusion import runner_process
import click

MODEL_CHOICES = ["v2.1-base"]
DEVICE_CHOICES = ["CPU FP32", "CPU/iGPU FP16"]

opt_model = None
opt_device = None
def update_options(local_model_path=None):
    '''
    when it's other occasions, will not set the option for users
    '''
    choices = []
    for model in MODEL_CHOICES:
        for device in DEVICE_CHOICES:
            if check_option(model, device, local_model_path):
                temp = model + " " + device
                if device == "CPU/iGPU FP16":
                    choices.append(f"{model} CPU FP16")
                    choices.append(f"{model} CPU + iGPU FP16")
                else:
                    choices.append(temp)
    if opt_model is not None and opt_device is not None:
        value = opt_model + " " + opt_device
    else:
        value = None
    return gr.update(value=value, choices=choices, label="switch option")

def pre_convert(model, device, token, local_model_path):
    '''
    align the input options from minimal UI with standard Web UI
    '''
    # now we only support Stable Diffusion v2.1-base so just depends on device
    if device == "CPU FP32":
        result = f"{model}/JIT"
    elif device == "CPU/iGPU FP16":
        result = f"{model}/OpenVINO FP16 iGPU"
    else:
        raise Exception(f"Not supported for device type: {device}")
    log = convert(result, token, local_model_path)
    global opt_model, opt_device
    opt_model, opt_device = model, device
    return log, update_options(local_model_path)


def pre_pipeline_process(model_dir, optimization_methods, scheduler, seed, prompt, scale, steps, height, width, neg_prompt):
    '''
    align the input options from minimal UI with standard Web UI
    '''
    model = optimization_methods.split(" ")[0]
    if "CPU FP32" in optimization_methods:
        result = f"{model}/JIT"
    elif "CPU FP16" in optimization_methods:
        result = f"{model}/OpenVINO FP16"
    elif "CPU + iGPU FP16" in optimization_methods:
        result = f"{model}/OpenVINO FP16 iGPU"
    else:
        raise Exception(f"not supported for {optimization_methods}")
    if runner_process.cpu_binding:
        fn = pipeline_process
        inputs = [model_dir, result, scheduler, seed, prompt, scale, steps, height, width, neg_prompt]
    else:
        fn = pipeline
        inputs = [model_dir, runner_process.var_dict, result, scheduler, seed, prompt, scale, steps, height, width, neg_prompt]
    return fn(*inputs)


def check_option(version, device, local_model_path=None):
    '''
    when local_model_path is set, will check this path instead of default "models" folder
    and will not use huggingface cache structure
    '''
    try:
        if local_model_path is None or local_model_path == "":
            local_model_path = local_model_path.strip('"').strip("'")
            model_dir = get_snapshot_dir_from_model_version(version)
        else:
            model_dir = local_model_path
        pipe = NanoStableDiffusionPipeline()
        input_accelerator = "openvino"
        input_device = "CPU"
        input_precision = "float16"

        if device == "CPU FP32":
            input_accelerator = "jit"
            input_precision = "float32"
        elif device == "CPU/iGPU FP16":
            input_device = "GPU"
            
        converted_dir = pipe._get_cache_path(model_dir, accelerator=input_accelerator, ipex=False, precision=input_precision, low_memory=False, device=input_device)
        vae_converted_dir = pipe._get_cache_path(model_dir, accelerator=input_accelerator, ipex=False, precision=input_precision, low_memory=False, device=input_device, vae=True)
        if os.path.exists(converted_dir) and len(os.listdir(converted_dir)) > 0 :            
            if vae_converted_dir:
                if os.path.exists(vae_converted_dir) and len(os.listdir(vae_converted_dir)) > 0:
                    return True
            else:
                return True
    except:
        return False
    return False


@click.command
@click.option('--cpu_binding', is_flag=True, help="Bind CPU cores")
def main(cpu_binding):
    if cpu_binding:
        runner_process.cpu_binding = True
    with gr.Blocks() as demo:
        with gr.Tab("optimize model"):
            Model = gr.Dropdown(value="v2.1-base", label="Model", interactive=True, choices=MODEL_CHOICES)
            Device = gr.Radio(value="CPU/iGPU FP16", label="Device", choices=DEVICE_CHOICES, interactive=True)
            local_model_path = gr.Textbox(label="local model path (optional)", value="")
            use_auth_token = gr.Textbox(label="huggingface token (optional)", value=None)
            
            output_log = gr.Textbox(label="Log")
            opt_btn = gr.Button("Optimize Model")            
            
        
        with gr.Tab("txt2img"):
            with gr.Row():
                with gr.Column():     
                    prompt = gr.Textbox(label="Prompt", value="A digital illustration of a medieval town, 4k, detailed, trending in art station, fantasy")
                    neg_prompt = gr.Textbox(label="Negative prompt")
                    with gr.Blocks():
                        scale = gr.Slider(0, 10, label="guidance scale", value=7.5, interactive=True)
                        steps = gr.Slider(0, 100, label="sampling steps", value=10, interactive=True)
                        scheduler = gr.Radio(["PNDMS", "DDIM", "Euler", "Euler A", "LMD", "DPM-Solver"], label="scheduler", value="DPM-Solver")
                        with gr.Row():
                            height = gr.Slider(1, 1024, label='height', value=512, interactive=True)
                            width = gr.Slider(1, 1024, label='width', value=512, interactive=True)
                        seed = gr.Number(interactive=True, precision=0, label='seed', value=-1)
                        
                        with gr.Row():
                            optimization_methods = gr.Dropdown(label="switch option")

                with gr.Column():
                    output = gr.Image(label="Image")
                    time_cost = gr.Text(label="Generation time (s)")
            gen_btn = gr.Button("Generate")

            
            gen_btn.click(
                fn=pre_pipeline_process, 
                inputs=[local_model_path, optimization_methods, scheduler, seed, prompt, scale, steps, height, width, neg_prompt], 
                outputs=[output, time_cost])
        opt_btn.click(
            fn=pre_convert,
            inputs=[Model, Device, use_auth_token, local_model_path],
            outputs=[output_log, optimization_methods] 
        )
        local_model_path.change(fn=update_options, inputs=[local_model_path], outputs=[optimization_methods])
        demo.load(fn=update_options, outputs=[optimization_methods])
    if runner_process.cpu_binding:
        p = psutil.Process()
        p.cpu_affinity([0])

    # set this if browser is not at local
    # demo.launch(share=True, server_name="0.0.0.0")
    
    demo.launch(share=True)

if __name__ == '__main__':
    main()
