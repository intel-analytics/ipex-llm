#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import ClassVar, Optional
import torch
import os
from bigdl.nano.pytorch import InferenceOptimizer

from diffusers.schedulers import *
from .utils import get_snapshot_dir_from_model_version
from .pipelines import NanoStableDiffusionPipeline

scheduler_map = {
    "DDIM": DDIMScheduler,
    "Euler A": EulerAncestralDiscreteScheduler,
    "Euler": EulerDiscreteScheduler,
    "LMD": LMSDiscreteScheduler,
    "PNDMS": PNDMScheduler,
    "DPM-Solver": DPMSolverMultistepScheduler
}



class StableDiffusionRunner:
    # uncomment the below class attributes loading to accelerate the test of the UI layout only
    
    _instance: ClassVar[Optional["StableDiffusionRunner"]] = None

    def __init__(self) -> None:        
        """
        Caution: All the attributes need to be set to serializable values
        because Python multiprocessing need to pickle the instance while starting new Process
        """
        self.scheduler = None # set a None to trigger scheduler initialization
        self.generator = None
        self.pipe = None
        self.model_dir = None

        self.calibrated = False
        self.version = None
        self.optimization_method = ""
        self.opt_postfix = "" # one optimization method could have multiple options
        self.device = None
        self.precision = None
        # model_id = "CompVis/stable-diffusion-v1-4"
        # TODO: may need a patch manager here
        # after applying the patch, need to re-import the module
    
    def init_scheduler(self, scheduler):
        if scheduler != self.scheduler or self.pipe.scheduler == None:
            local_scheduler = os.path.join(self.model_dir, "scheduler")
            self.pipe.switch_scheduler(scheduler, local_scheduler)
            # scheduler_cls = scheduler_map[scheduler]
            # # modify the line below if you want to download the scheduler config
            # local_scheduler = os.path.join(self.model_dir, "scheduler")
            # if os.path.isdir(local_scheduler):
            #     self.d_scheduler = scheduler_cls.from_pretrained(local_scheduler)
            # else:
            #     raise Exception("No scheduler config found in directory")
            # self.scheduler = scheduler

    def init_generator(self, seed):
        if self.generator is None:
            self.generator = torch.Generator()       
        if seed != -1:
            self.generator.manual_seed(seed)
        else:
            self.generator.seed()
        
    def need_init_pipeline(self, version, device, precision, optimization_method, opt_postfix=""):
        r"""
        check if the pipeline need to be reloaded and set current corresponding status

        Args:
            optimization_method:
                the optimization method, could be nano_jit, nano_openvino, bes_openvino
            opt_postfix:
                the config of this optimization method, e.g. nano_jit could have ipex enabled/disabled
        """
        if self.optimization_method + self.opt_postfix != optimization_method + opt_postfix or self.version != version or self.device != device or self.precision != precision:
            res = True
            self.pipe = None # free memory
            print(f"Switching pipeline from {self.version}/{self.precision}/{self.device}/{self.optimization_method}/{self.opt_postfix} to {version}/{precision}/{device}/{optimization_method}/{opt_postfix}")
        elif self.pipe is None:
            res = True
        else:
            res = False
        # record current pipeline config
        self.optimization_method = optimization_method
        self.opt_postfix = opt_postfix
        self.version, self.device, self.precision = version, device, precision 
        return res

    @classmethod
    def initialize(cls):
        if StableDiffusionRunner._instance is None:
            print("Initializing StableDiffusionRunner...")
            StableDiffusionRunner._instance = cls()
        return StableDiffusionRunner._instance
        # os.environ["NANO_HOME"] = f"{APP_ROOT}/nano_stable_diffusion2"
        # StableDiffusionRunner._instance.model_dir = os.environ["NANO_HOME"]

    @staticmethod
    def run(model_dir, optimization_methods, scheduler, **kwargs):
        assert StableDiffusionRunner._instance is not None
        runner = StableDiffusionRunner._instance
        return runner._run(model_dir, optimization_methods, scheduler, **kwargs)
    
    def _run(self, model_dir, optimization_methods, scheduler, seed, **kwargs):
        self.init_generator(seed)
        image = self.run_opt_pipeline(
            model_dir, 
            scheduler=scheduler,
            optimization_methods=optimization_methods,
            generator=self.generator,
            **kwargs)
        return image

    def run_opt_pipeline(self, model_dir, optimization_methods, scheduler, **kwargs):
        version, optimization_methods = optimization_methods.split('/')
        if model_dir is not None and model_dir != "":
            self.model_dir = model_dir
        else:
            self.model_dir = get_snapshot_dir_from_model_version(version)
        
        # fp16 only works with OpenVINO on iGPU
        if "FP16" in optimization_methods:
            precision = "float16"
        elif "INT8" in optimization_methods:
            precision = "int8"
        else:
            precision = "float32"
        device = "GPU" if "iGPU" in optimization_methods else "CPU"

        if "OpenVINO" in optimization_methods:
            optimization_methods = "OpenVINO"

        if optimization_methods == "IPEX":
            image = self.run_ipex_jit(scheduler, version, precision, device, **kwargs)
        elif optimization_methods == "JIT":
            image = self.run_jit(scheduler, version, precision, device, **kwargs)
        elif optimization_methods == "INC INT8":
            image = self.run_inc_int8(scheduler, version, precision, device, **kwargs)
        elif optimization_methods == "OpenVINO":
            image = self.run_ov(scheduler, version, precision, device, **kwargs)
        elif optimization_methods == "IPEX Low-memory":
            image = self.run_ipex_jit_lm(scheduler, version, precision, device, **kwargs)
        else:
            raise Exception(f"Optimization method not supported, supported methods are 'IPEX', 'IPEX Low-memory', 'OpenVINO', "
                             "'INC INT8', got {optimization_methods}")

        return image

   
    def run_ipex_jit(self, scheduler, version, precision, device, **kwargs):        
        # set_env_var(IPEX_ENV_VAR)
        # calibration

        if self.need_init_pipeline(version, device, precision, "nano_jit", "ipex"):
            self.pipe = NanoStableDiffusionPipeline.from_pretrained(
                pretrained_model_path=self.model_dir, 
                accelerator="jit", 
                precision=precision, 
                device=device)
        self.init_scheduler(scheduler)
        # self.pipe.scheduler = self.d_scheduler
        with InferenceOptimizer.get_context(self.pipe.unet):
            image = self.pipe(**kwargs)[0]
        return image

    def run_ipex_jit_lm(self, scheduler, version, precision, device, **kwargs):
        # run low-memory version ipex
        # set_env_var(IPEX_ENV_VAR)
        if self.need_init_pipeline(version, device, precision, "nano_jit", "ipex_low_memory"):            
            self.pipe = NanoStableDiffusionPipeline.from_pretrained(
                pretrained_model_path=self.model_dir,
                accelerator="jit",
                low_memory=True, 
                precision=precision, 
                device=device)
        self.init_scheduler(scheduler)
        with InferenceOptimizer.get_context(self.pipe.unet):
            image = self.pipe(**kwargs)[0]
        return image

    def run_jit(self, scheduler, version, precision, device, **kwargs):        
        # set_env_var(IPEX_ENV_VAR)
        # calibration
        if self.need_init_pipeline(version, device, precision, "nano_jit"):            
            self.pipe = NanoStableDiffusionPipeline.from_pretrained(
                pretrained_model_path=self.model_dir, 
                accelerator="jit",
                ipex=False, 
                precision=precision, 
                device=device)
        self.init_scheduler(scheduler)
        with InferenceOptimizer.get_context(self.pipe.unet):
            image = self.pipe(**kwargs)[0]
        return image    

    def run_ov(self, scheduler, version, precision, device, **kwargs):
        # set_env_var(OV_ENV_VAR)
        # calibration
        if self.need_init_pipeline(version, device, precision, "nano_openvino"):
            self.pipe = NanoStableDiffusionPipeline.from_pretrained(
                pretrained_model_path=self.model_dir, 
                accelerator="openvino", 
                precision=precision, 
                device=device)
        self.init_scheduler(scheduler)
        with InferenceOptimizer.get_context(self.pipe.unet):
            image = self.pipe(**kwargs)[0]
        return image
    
    def run_inc_int8(self, scheduler, version, precision, device, **kwargs):
        if self.need_init_pipeline(version, device, precision, "nano_inc_int8"):
            self.pipe = NanoStableDiffusionPipeline.from_pretrained(
                pretrained_model_path=self.model_dir, 
                accelerator=None,
                ipex=False,
                precision=precision, 
                device=device)
        self.init_scheduler(scheduler)
        with InferenceOptimizer.get_context(self.pipe.unet):
            image = self.pipe(**kwargs)[0]
        return image
