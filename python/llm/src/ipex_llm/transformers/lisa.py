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

from transformers import TrainerCallback
import numpy as np
from ipex_llm.utils.common import invalidInputError


# source: https://github.com/OptimalScale/LMFlow/blob/main/src/lmflow/pipeline/finetuner.py
class DynamicLayerActivationCallback(TrainerCallback):
    def __init__(self, n_layers, interval_steps, model):
        super().__init__()
        self.n_layers = n_layers
        self.interval_steps = interval_steps
        self.model = model

        # Determine the way to access layers based on the model type
        class_to_layers_map = {
            'LlamaForCausalLM': 'model.model.layers',
            'Qwen2ForCausalLM': 'model.model.layers',
            'MistralForCausalLM': 'model.model.layers',
            'MixtralForCausalLM': 'model.model.layers',
            'GemmaForCausalLM': 'model.model.layers',
            'GPT2LMHeadModel': 'model.transformer.h',
            'ChatGLMModel': 'model.transformer.encoder.layers',
        }
        model_class_name = self.model.__class__.__name__
        if model_class_name in class_to_layers_map:
            self.layers_attribute = class_to_layers_map[model_class_name]
        else:
            # self.layers_attribute = training_args.lisa_layers_attribute
            invalidInputError(False, f"Model {model_class_name} not supported.")
        # Dynamically execute to get the number of layers
        self.total_layers = len(eval('self.' + self.layers_attribute))

        self.active_layers_indices = []

    def freeze_all_layers(self):
        layers = eval('self.' + self.layers_attribute)  # Dynamically execute to get layers
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False

    def on_step_begin(self, args, state, control, **kwargs):
        # Check if it's time to switch active layers, including at step 0
        if state.global_step % self.interval_steps == 0:
            self.switch_active_layers()

    def switch_active_layers(self):
        # First, disable gradients for all layers
        self.freeze_all_layers()

        # Randomly select n_layers to activate
        layers = eval('self.' + self.layers_attribute)  # Re-fetch layer references
        self.active_layers_indices = np.random.choice(
            range(self.total_layers),
            self.n_layers,
            replace=False
        )
        print(
            f"Activating layers at indices: {self.active_layers_indices} for the next steps.",
            flush=True
        )

        # Enable gradients only for the selected layers
        for idx in self.active_layers_indices:
            for param in layers[idx].parameters():
                param.requires_grad = True
