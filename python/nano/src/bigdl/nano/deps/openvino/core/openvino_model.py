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
from openvino.runtime import Core
from openvino.runtime.passes import Manager
from bigdl.nano.utils.log4Error import invalidInputError
from openvino.tools.pot.graph import load_model, save_model
from openvino.tools.pot.engines.ie_engine import IEEngine
from openvino.tools.pot.pipeline.initializer import create_pipeline
from openvino.tools.pot.graph.model_utils import compress_model_weights


class OpenVINOModel:
    def __init__(self, ie_network: str):
        self.ie_network = None
        self.read_network(ie_network)

    def forward_step(self, *inputs):
        return self.infer_request.infer(list(inputs))

    def read_network(self, model: str):
        core = Core()
        self.ie_network = core.read_model(model=model)
        self.exec_model = core.compile_model(model=self.ie_network, device_name='CPU')
        self.infer_request = self.exec_model.create_infer_request()

    def _save_model(self, path):
        """
        Save PytorchOpenVINOModel to local as xml and bin file

        :param path: Path to save the model.
        """
        path = Path(path)
        invalidInputError(self.ie_network,
                          "self.ie_network shouldn't be None.")
        invalidInputError(path.suffix == ".xml",
                          "Path of openvino model must be with '.xml' suffix.")
        path.mkdir(exist_ok=True)
        xml_path = path / self.status['xml_path']
        pass_manager = Manager()
        pass_manager.register_pass(pass_name="Serialize",
                                   xml_path=str(xml_path),
                                   bin_path=str(xml_path.with_suffix(".bin")))
        pass_manager.run_passes(self.ie_network)

    def pot(self,
            dataloader,
            metric=None,
            drop_type="relative",
            maximal_drop=0.999,
            max_iter_num=1,
            n_requests=None,
            sample_size=300):
        # pot has its own model format, so we need to save and reload by pot
        with TemporaryDirectory() as dir:
            dir = Path(dir)
            self._save_model(dir)
            model_config = {
                "model_name": "model",
                "model": str(dir / 'ov_saved_model.xml'),
                "weights": str(dir / 'ov_saved_model.bin')
            }
            model = load_model(model_config)
            engine_config = {"device": "CPU",
                             "stat_requests_number": n_requests,
                             "eval_requests_number": n_requests}
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
            engine = IEEngine(config=engine_config, data_loader=dataloader, metric=metric)
            pipeline = create_pipeline(algorithms, engine)
            compressed_model = pipeline.run(model=model)
            compress_model_weights(model=compressed_model)
            # To use runtime, we need to save and reload
            # returned a list of paths
            dir = "optimized_model"
            compressed_model_paths = save_model(
                model=compressed_model,
                save_path=dir,
                model_name='model'
            )
            return compressed_model_paths
