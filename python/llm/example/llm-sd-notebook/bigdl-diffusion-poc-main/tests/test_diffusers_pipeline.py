import unittest
import os
import sys
from diffusers import *
from PIL import Image

from bigdl_diffusion.diffusers.pipelines import *
from tests import get_resource, get_output

class TestPipeline(unittest.TestCase):
    def test_original_txt2img_pipeline(self):
        pipe = StableDiffusionPipeline.from_pretrained(os.getenv('TEST_MODEL_PATH'))
        output = pipe("A digital illustration of a medieval town, 4k, detailed, trending in art station, fantasy", num_inference_steps=10).images[0]
        output.save(get_output('castle_original.png'))

    def test_nano_txt2img_pipeline_ov(self):
        pipe = NanoStableDiffusionPipeline.from_pretrained(os.getenv('TEST_MODEL_PATH'), device='CPU', backend='ov')
        output = pipe("A digital illustration of a medieval town, 4k, detailed, trending in art station, fantasy", num_inference_steps=10).images[0]
        output.save(get_output('castle_nano.png'))

    def test_nano_txt2img_pipeline_ipex(self):
        pipe = NanoStableDiffusionPipeline.from_pretrained(os.getenv('TEST_MODEL_PATH'), device='CPU', backend='ipex', precision='bfloat16')
        output = pipe("A digital illustration of a medieval town, 4k, detailed, trending in art station, fantasy", num_inference_steps=10).images[0]
        output.save('castle_nano.png')
        
    def test_original_img2img_pipeline(self):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(os.getenv('TEST_MODEL_PATH'))
        image = Image.open(get_resource('flower_in_vase.png'))
        output = pipe("paper flowers in vase, light pink background", image=image, num_inference_steps=10).images[0]
        output.save(get_output('paper_flower_original.png'))

    def test_nano_img2img_pipeline_ov(self):
        pipe = NanoStableDiffusionImg2ImgPipeline.from_pretrained(os.getenv('TEST_MODEL_PATH'), device='CPU')
        image = Image.open(get_resource('flower_in_vase.png'))
        output = pipe("paper flowers in vase, light pink background", image=image, num_inference_steps=10).images[0]
        output.save(get_output('paper_flower_nano.png'))
        
    def test_nano_img2img_pipeline_ipex(self):
        pipe = NanoStableDiffusionImg2ImgPipeline.from_pretrained(os.getenv('TEST_MODEL_PATH'), device='CPU', backend='ipex', precision='bfloat16')
        image = Image.open(get_resource('flower_in_vase.png'))
        output = pipe("paper flowers in vase, light pink background", image=image, num_inference_steps=10).images[0]
        output.save(get_output('paper_flower_original.png'))
        

    

if __name__ == '__main__':
    unittest.main()