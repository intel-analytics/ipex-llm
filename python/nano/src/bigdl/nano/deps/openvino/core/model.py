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
from bigdl.nano.utils.common import invalidInputError, _flatten, invalidOperationError
from openvino.runtime import Model
from openvino.runtime import AsyncInferQueue
import numpy as np
from datetime import datetime
from .utils import save, OpenVINO_LESS_2022_3


class OpenVINOModel:
    def __init__(self, ie_network: str, device='CPU', precision='fp32',
                 thread_num=None, config=None, shapes=None):
        self._ie = Core()
        # check device
        self._check_device(self._ie, device)
        self._device = device
        self._precision = precision
        self.thread_num = thread_num
        self.additional_config = config
        self.shapes = shapes
        self.ie_network = ie_network

    def on_forward_start(self, inputs):
        self._model_exists_or_err()
        return inputs

    def forward_step(self, *inputs, **kwargs):
        flattened_inputs = []
        _flatten(inputs, flattened_inputs)
        args_length = len(inputs)
        if kwargs is not None and len(kwargs) > 0:
            # check kwargs first, to avoid user call model(t1, t2, b=t3, c=t4)
            # when the signature is def forward(a, b, c)
            for arg in kwargs:
                if arg in self.forward_args[:args_length]:
                    invalidInputError(False,
                                      f"You shouldn't pass arguement {arg} as it has "
                                      "been passed as a positional arguement.")
            # add inputs based on original order
            for arg in self.forward_args[args_length:]:
                if arg in kwargs:
                    if isinstance(kwargs[arg], (list, tuple)):
                        flattened_inputs.extend(kwargs[arg])
                    else:
                        flattened_inputs.append(kwargs[arg])
        if len(self._forward_args) != len(flattened_inputs):
            # formatting a Tensor will cost much time,
            # so we put it in this `if` statement
            invalidInputError(False,
                              "The length of inputs is "
                              "inconsistent with the length of OpenVINO's inputs, "
                              f"got model_forward_args: {self._forward_args}, "
                              f"and flattened inputs's length: {len(flattened_inputs)}")
        return self._infer_request.infer(list(flattened_inputs))

    def on_forward_end(self, outputs):
        arrays = tuple(outputs.values())
        if len(arrays) == 1:
            arrays = arrays[0]
        return arrays

    def forward(self, *inputs, **kwargs):
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
            if OpenVINO_LESS_2022_3:
                config = {"CPU_THREADS_NUM": str(self.thread_num)}
            else:
                config = {"INFERENCE_NUM_THREADS": str(self.thread_num)}
        else:
            config = {}
        if self.additional_config is not None:
            if self._device == 'CPU':
                # TODO: check addition config based on device
                config.update(self.additional_config)
            else:
                ov_cache_dir = self.additional_config.get("CACHE_DIR", None)
                if ov_cache_dir:
                    config["CACHE_DIR"] = ov_cache_dir
        self.final_config = config

        if self.shapes is None:
            self.create_infer_request()
        else:
            self._infer_request = None
            self.reshape(self.shapes)

        input_names = [t.any_name for t in self._ie_network.inputs]
        self._forward_args = input_names

    def create_infer_request(self):
        start_time = datetime.utcnow()
        self._compiled_model = self._ie.compile_model(model=self.ie_network,
                                                      device_name=self._device,
                                                      config=self.final_config)
        self._infer_request = self._compiled_model.create_infer_request()
        duration_ms = f"{(datetime.utcnow() - start_time).total_seconds() * 1000:.2f}"
        print(f"Compile model and create infer request took {duration_ms} ms")

    def reshape(self, shapes):
        """
        Reshape the model to fit the inputs.Be aware that not all models support reshaping,
        and models that do, may not support all input shapes. The model accuracy may also
        suffer if you reshape the model.
        :param shapes: input shape. For example, 'input1[1,3,224,224],input2[1,4]', '[1,3,224,224]'.
               This parameter affect model Parameter shape, can be dynamic. For dynamic dimesions
               use symbol `?`, `-1` or range `low.. up`.'
        """
        invalidInputError(isinstance(shapes, str), "Shapes only supports string inputs "
                          "like 'input1[1,3,224,224],input2[1,4]', '[1,3,224,224]' but got "
                          f"{shapes.__class__.__name__}.")

        success = True
        error_msg = None
        shapes, reshape, origin_shapes = self._get_reshape_info(shapes)
        if not reshape:
            print("Skip the reshape process since the input shapes are same as the current "
                  "model shapes.")
            if self._infer_request:
                print("Skip compiling model.")
                return success, error_msg
        else:
            start_time = datetime.utcnow()
            print('Reshaping model: {}'
                  .format(', '.join("'{}': {}".format(k, str(v)) for k, v in shapes.items())))
            try:
                self.ie_network.reshape(shapes)
                duration_ms = f"{(datetime.utcnow() - start_time).total_seconds() * 1000:.2f}"
                print(f"Reshape model took {duration_ms} ms")
            except Exception as e:
                success = False
                error_msg = f"Failed to reshape this model. Error message: \n {str(e)}"
                if self._infer_request:
                    print(error_msg)
                    return success, error_msg
                else:
                    error_msg += "\nCompile the original model instead."
                print(error_msg)

        try:
            self.create_infer_request()
        except RuntimeError as e:
            success = False
            error_msg = f"Failed to compile the reshaped model. Error message: \n {str(e)}"
            invalidOperationError(self._infer_request is not None, error_msg)
            # Reshape to the original shapes
            print(error_msg)
            print("Revert the reshaping process.")
            self.ie_network.reshape(origin_shapes)

        return success, error_msg

    def _get_reshape_info(self, shapes):
        invalidInputError(isinstance(shapes, str),
                          "`_get_reshape_info` only supports string input.")
        from openvino.tools.benchmark.utils.utils import parse_input_parameters
        from openvino.runtime import PartialShape

        inputs = self.ie_network.inputs
        input_names = [port.any_name for port in inputs]
        inputs_info = [(i.any_name, i.node.friendly_name, i.partial_shape) for i in inputs]
        shape_map = parse_input_parameters(shapes, input_names=input_names)
        reshape = False
        input_shapes = {}
        origin_shapes = {}
        for name, node_name, shape in inputs_info:
            origin_shapes[name] = shape
            new_shape = None
            if name in shape_map:
                new_shape = PartialShape(shape_map[name])
            elif node_name in shape_map:
                new_shape = PartialShape(shape_map[node_name])

            if new_shape is None:
                input_shapes[name] = shape
            else:
                if new_shape != shape:
                    reshape = True
                input_shapes[name] = new_shape

        return input_shapes, reshape, origin_shapes

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
