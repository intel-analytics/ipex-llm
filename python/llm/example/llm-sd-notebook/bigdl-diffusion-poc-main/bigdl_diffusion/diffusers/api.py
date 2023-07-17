import click
from diffusers import StableDiffusionPipeline

def generate():
    pipe = StableDiffusionPipeline()


@click.command()
@click.option('-m', '--model_path', type=str)
@click.option('--prompt', type=str)
def main(model_path, prompt):
    generate()


if __name__ == '__main__':
    main()