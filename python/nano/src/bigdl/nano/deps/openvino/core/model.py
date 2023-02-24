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
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Union  # for typehint
from openvino.runtime import Core
from bigdl.nano.utils.common import invalidInputError
from openvino.runtime import Model
from .utils import save
from openvino.runtime import AsyncInferQueue
import numpy as np


class OpenVINOModel:
    def __init__(self, ie_network: str, device='CPU', precision='fp32',
                 thread_num=None, config=None):
        self._ie = Core()
        # check device
        self._check_device(self._ie, device)
        self._device = device
        self._precision = precision
        self.thread_num = thread_num
        self.additional_config = config
        self.ie_network = ie_network

    def on_forward_start(self, inputs):
        self._model_exists_or_err()
        return inputs

    def forward_step(self, *inputs):
        return self._infer_request.infer(list(inputs))

    def on_forward_end(self, outputs):
        arrays = tuple(outputs.values())
        if len(arrays) == 1:
            arrays = arrays[0]
        return arrays

    def forward(self, *inputs):
        inputs = self.on_forward_start(inputs)
        outputs = self.forward_step(*inputs)
        return self.on_forward_end(outputs)

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def _check_device(self, ie, device):
        devices = ie.available_devices
        if device == 'GPU' and 'GPU.0' in devices:
            # GPU is equivalent to GPU.0
            return True
        invalidInputError(device in devices,
                          "Your machine don't have {} device (only have {}), please modify "
                          "the incoming device value.".format(device, ",".join(list(devices))))

    @property
    def forward_args(self):
        return self._forward_args

    @property
    def ie_network(self):
        return self._ie_network

    @ie_network.setter
    def ie_network(self, model):
        if isinstance(model, (str, Path)):
            self._ie_network = self._ie.read_model(model=str(model))
        else:
            self._ie_network = model
        if self.thread_num is not None and self._device == 'CPU':
            config = {"CPU_THREADS_NUM": str(self.thread_num)}
        else:
            config = {}
        if self.additional_config is not None and self._device == 'CPU':
            # TODO: check addition config based on device
            config.update(self.additional_config)
        self._compiled_model = self._ie.compile_model(model=self.ie_network,
                                                      device_name=self._device,
                                                      config=config)
        self._infer_request = self._compiled_model.create_infer_request()
        self.final_config = config
        input_names = [t.any_name for t in self._ie_network.inputs]
        self._forward_args = input_names

    def _save(self, path):
        """
        Save OpenVINOModel to local as xml and bin file

        :param path: Directory to save the model.
        """
        self._model_exists_or_err()
        path = Path(path)
        path.mkdir(exist_ok=True)
        xml_path = path / 'ov_saved_model.xml'
        save(self.ie_network, xml_path)

    def pot(self,
            dataloader,
            metric=None,
            drop_type="relative",
            maximal_drop=0.999,
            max_iter_num=1,
            n_requests=None,
            sample_size=300) -> Model:
        from openvino.tools.pot.graph import load_model, save_model
        from openvino.tools.pot.engines.ie_engine import IEEngine
        from openvino.tools.pot.pipeline.initializer import create_pipeline
        from openvino.tools.pot.graph.model_utils import compress_model_weights
        # set batch as 1 if it's dynaminc or larger than 1
        orig_shape = dict()
        static_shape = dict()
        for i, input_obj in enumerate(self.ie_network.inputs):
            orig_shape[i] = input_obj.get_partial_shape()
            shape = input_obj.get_partial_shape()
            # modify dynamic axis to 1 if it's batch dimension
            shape[0] = 1
            static_shape[i] = shape
        self.ie_network.reshape(static_shape)

        # pot has its own model format, so we need to save and reload by pot
        with TemporaryDirectory() as root:
            dir = Path(root)
            save(self.ie_network, str(dir / 'model.xml'))

            # Convert model back to original shape
            self.ie_network.reshape(orig_shape)

            model_config = {
                "model_name": "model",
                "model": str(dir / 'model.xml'),
                "weights": str(dir / 'model.bin')
            }
            model = load_model(model_config)

        engine_config = {"device": "CPU",
                         "stat_requests_number": n_requests,
                         "eval_requests_number": n_requests}
        engine = IEEngine(config=engine_config, data_loader=dataloader, metric=metric)

        algorithms = [
            {
                "name": "DefaultQuantization",
                "params": {
                    "target_device": "CPU",
                    "preset": "performance",
                    "stat_subset_size": sample_size,
                },
            }
        ]
        if metric:
            algorithms = [
                {
                    "name": "AccuracyAwareQuantization",
                    "params": {
                        "target_device": "CPU",
                        "preset": "performance",
                        "stat_subset_size": sample_size,
                        "maximal_drop": maximal_drop,
                        "max_iter_num": max_iter_num,
                        "drop_type": drop_type,
                    },
                }
            ]

        pipeline = create_pipeline(algorithms, engine)
        compressed_model = pipeline.run(model=model)
        compress_model_weights(model=compressed_model)

        # To use runtime, we need to save and reload
        # returned a list of paths, but for now there is only one model path in list
        with TemporaryDirectory() as root:
            compressed_model_paths = save_model(
                model=compressed_model,
                save_path=root,
                model_name='model'
            )
            # set batch for compressed model
            model_path = compressed_model_paths[0]['model']
            model = Core().read_model(model_path)
            model.reshape(orig_shape)
        return model

    def _model_exists_or_err(self):
        invalidInputError(self.ie_network is not None, "self.ie_network shouldn't be None.")

    def async_predict(self,
                      input_data: Union[List[np.ndarray], List[List[np.ndarray]]],
                      num_requests: int = 0) -> List[np.ndarray]:
        """
        Perfrom model inference using async mode.

        :param input_data: Input data to be inferenced.
                           Users can put multiple input data in a list to infer them together.
                           Can be List[numpy.ndarray] or
                           List[List[numpy.ndarray]] if the model has multiple inputs.
        :param num_requests: Number of requests in the asynchronous infer requests pool.
                             Each element in input_data will be bound to an idle async
                             infer request in the pool to do inference.
                             Defaults to 0.
                             If 0, it will be set automatically to the optimal number.

        :return: A List containing result of each inference. Type: List[numpy.ndarray]
        """
        results = [0 for _ in range(len(input_data))]

        # call back function is called when a infer_request in the infer_queue
        # finishes the inference
        def call_back(requests, idx):
            results[idx] = self.on_forward_end(requests.results)

        infer_queue = AsyncInferQueue(self._compiled_model, jobs=num_requests)
        infer_queue.set_callback(call_back)

        for id, model_input in enumerate(input_data):
            if isinstance(model_input, np.ndarray):
                # start_async accpet a list of input data
                # so put data in a list
                model_input = [model_input]
            infer_queue.start_async(model_input, userdata=id)

        infer_queue.wait_all()

        return results
