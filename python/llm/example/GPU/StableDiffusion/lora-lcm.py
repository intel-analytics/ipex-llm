import torch
from diffusers import DiffusionPipeline, LCMScheduler
import ipex_llm
import argparse


def main(args):
    pipe = DiffusionPipeline.from_pretrained(
        args.repo_id_or_model_path,
        torch_dtype=torch.bfloat16,
    ).to("xpu")

    # set scheduler
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # load LCM-LoRA
    pipe.load_lora_weights(args.lora_weights_path)

    generator = torch.manual_seed(42)
    image = pipe(
        prompt=args.prompt, num_inference_steps=args.num_steps, generator=generator, guidance_scale=1.0
    ).images[0]
    image.save("lcm-lora-sdxl-gpu.png")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Stable Diffusion lora-lcm")
    parser.add_argument('--repo-id-or-model-path', type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                        help='The huggingface repo id for the stable diffusion model checkpoint')
    parser.add_argument('--lora-weights-path',type=str,default="latent-consistency/lcm-lora-sdxl",
                        help='The huggingface repo id for the lcm lora sdxl checkpoint')
    parser.add_argument('--prompt', type=str, default="A lovely dog on the table, detailed, 8k",
                        help='Prompt to infer')
    parser.add_argument('--save-path',type=str,default="lcm-lora-sdxl-gpu.png",
                        help="Path to save the generated figure")
    parser.add_argument('--num-steps',type=int,default=4,
                        help="Number of inference steps")
    args = parser.parse_args()
    main(args)