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


from .version import TORCH_VERSION_LESS_1_10
from .version import TORCH_VERSION_LESS_1_11
from .version import TORCH_VERSION_LESS_1_12
from .version import TORCH_VERSION_LESS_1_13
from .version import TORCHVISION_VERSION_LESS_1_12
from .version import TORCHVISION_VERSION_LESS_1_14
from .version import TORCH_VERSION_LESS_2_0

from .attributes import patch_attrs_from_model_to_object

from .check_deps import check_ccl

from .channel_last import ChannelsLastCallback
from .channel_last import generate_channels_last_available
from .channel_last import apply_proper_channels_last

from .dataset import RepeatDataset
from .dataset import remove_batch_dim_fn

from .metric import NanoMetric

from .inspect import get_forward_args
from .inspect import get_forward_defaults
from .inspect import get_forward_annotations
from .inspect import get_conditional_args

from .dataloader import transform_multiple_input_dataloader_to_inc_mode
from .dataloader import automatic_add_label_in_dataloader

from .input_sample import get_input_example
from .input_sample import complement_input_sample

from .convert import export_to_onnx

from .save import save_model
from .save import transform_state_dict_to_dtype
from .load import load_model

from .xpu import apply_data_to_xpu, apply_data_to_half

from .jit_method import jit_convert

from .metadata import MetaData
