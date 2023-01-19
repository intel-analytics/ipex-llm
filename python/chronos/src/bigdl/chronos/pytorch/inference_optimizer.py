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

from bigdl.nano.pytorch import InferenceOptimizer as TSInferenceOptimizer

class InferenceOptimizer(TSInferenceOptimizer):

    def optimize(self, model: nn.Module,
                 training_data: Union[DataLoader, torch.Tensor, Tuple[torch.Tensor]],
                 validation_data: Optional[Union[DataLoader, torch.Tensor, Tuple[torch.Tensor]]] = None,
                 input_sample: Union[torch.Tensor, Dict, Tuple[torch.Tensor], None] = None,
                 metric: Optional[Callable] = None, direction: str = "max", thread_num: Optional[int] = None,
                 accelerator: Optional[Tuple[str]] = None,
                 precision: Optional[Tuple[str]] = None,
                 use_ipex: Optional[bool] = None,
                 search_mode: str = "default",
                 dynamic_axes: Union[bool, dict] = True,
                 logging: bool = False,
                 latency_sample_num: int = 100,
                 includes: Optional[List[str]] = None,
                 excludes: Optional[List[str]] = None,
                 output_filename: Optional[str] = None) -> None:

        super().optimize(model, training_data, validation_data,
                         input_sample, metric, direction,
                         thread_num, accelerator, precision,
                         use_ipex, search_mode, dynamic_axes,
                         logging, latency_sample_num,
                         includes, excludes, output_filename)

