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
from typing import Sequence, Any, Union, Dict

import yaml
import tensorflow as tf

from bigdl.nano.utils.common import invalidInputError


class KerasOptimizedModel(tf.keras.Model):
    """A base class for keras optimized model."""

    def __call__(self, *args, **kwargs):
        """Run inference, automatically perform type and dtype conversion."""
        kwargs.pop("training", None)
        inputs = self.preprocess(args, kwargs)
        outputs = self.forward(inputs)
        outputs = self.postprocess(outputs)
        return outputs

    def call(self, *args, **kwargs):
        """The same as __call__."""
        return self(*args, **kwargs)

    def preprocess(self, args: Sequence[Any], kwargs: Dict[str, Any]):
        """Preprocess inputs, such as convert inputs to numpy ndarray."""
        invalidInputError(False, "preprocess function is not implemented.")

    def forward(self, inputs: Union[Sequence[Any], Dict[str, Any]]):
        """Run inference."""
        invalidInputError(False, "forward function is not implemented.")

    def postprocess(self, outputs: Sequence[Any]):
        """Postprocess outputs, such as convert outputs to tensorflow Tensor."""
        return outputs

    def predict(self,
                x,
                batch_size=None,
                verbose='auto',
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False):
        """Same to keras model's `predict` method, except that it runs in eager mode."""
        # todo: we should implement our predict method, because tf's predict will convert
        # numpy to tensor, then convert tensor to numpy when using ov/onnx optimized model
        self.run_eagerly = True
        return super().predict(x=x,
                               batch_size=batch_size,
                               verbose=verbose,
                               steps=steps,
                               callbacks=callbacks,
                               max_queue_size=max_queue_size,
                               workers=workers,
                               use_multiprocessing=use_multiprocessing)

    def evaluate(self,
                 x=None,
                 y=None,
                 batch_size=None,
                 verbose='auto',
                 sample_weight=None,
                 steps=None,
                 callbacks=None,
                 max_queue_size=10,
                 workers=1,
                 use_multiprocessing=False,
                 return_dict=False,
                 **kwargs):
        """Same to keras model's `evaluate` method, except that it runs in eager mode."""
        # todo: we should implement our evaluate method, because tf's predict will convert
        # numpy to tensor, then convert tensor to numpy when using ov/onnx optimized model
        self.run_eagerly = True
        return super().evaluate(x=x,
                                y=y,
                                batch_size=batch_size,
                                verbose=verbose,
                                sample_weight=sample_weight,
                                steps=steps,
                                callbacks=callbacks,
                                max_queue_size=max_queue_size,
                                workers=workers,
                                use_multiprocessing=use_multiprocessing,
                                return_dict=return_dict,
                                **kwargs)

    @property
    def status(self):
        """Return model's status."""
        return {"ModelType": type(self).__name__}

    def _dump_status(self, path):
        meta_path = Path(path) / "nano_model_meta.yml"
        with open(meta_path, 'w') as f:
            yaml.safe_dump(self.status, f)

    @staticmethod
    def _load_status(path):
        meta_path = Path(path) / "nano_model_meta.yml"
        with open(meta_path, 'r') as f:
            metadata = yaml.safe_load(f)
        return metadata

    def _save(self, path, compression="fp32"):
        invalidInputError(False, "Saving function is not implemented.")

    @staticmethod
    def _load(path, model=None):
        invalidInputError(False, "Loading function is not implemented.")
