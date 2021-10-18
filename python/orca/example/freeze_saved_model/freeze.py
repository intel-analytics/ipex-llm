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

from optparse import OptionParser
import sys
import tensorflow as tf

from bigdl.dllib.utils.tf import export_tf

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--saved_model_path", dest="saved_model_path",
                      help="The path to a TensorFlow saved model")
    parser.add_option("--tag", default="serve", dest="tag",
                      help="The tag used to load from saved model")
    parser.add_option("--signature", default="serving_default", dest="signature",
                      help="The signature to find input and output tensors")
    parser.add_option("--input_tensors", default=None, dest="input_tensors",
                      help="A comma separated list of Tensor" +
                           " names as the model inputs, e.g. input_0:0,input_1:0."
                           " This will override the ones found in signature.")
    parser.add_option("--output_tensors", default=None, dest="output_tensors",
                      help="A comma separated list of Tensor" +
                           " names as the model outputs, e.g. output_0:0,output_1:0."
                           " This will override the ones found in signature.")
    parser.add_option("--output_path", dest="output_path",
                      help="The output frozen model path")
    (options, args) = parser.parse_args(sys.argv)

    assert options.saved_model_path is not None, "--saved_model_path must be provided"
    assert options.output_path is not None, "--output_path must be provided"

    with tf.Session() as sess:
        loaded = tf.saved_model.load(sess, tags=[options.tag],
                                     export_dir=options.saved_model_path)

        signature = loaded.signature_def[options.signature]
        if options.input_tensors is None:
            input_keys = signature.inputs.keys()
            input_names = [signature.inputs[key].name for key in input_keys]
            print("Found inputs in signature {}".format(options.signature))
            print("Inputs are \n{}".format(signature.inputs))
        else:
            input_names = options.input_tensors.split(",")
        input_tensors = [tf.get_default_graph().get_tensor_by_name(name) for name in input_names]

        if options.output_tensors is None:
            outputs_keys = signature.outputs.keys()
            output_names = [signature.outputs[key].name for key in outputs_keys]
            print("Found outputs in signature {}".format(options.signature))
            print("outputs are \n{}".format(signature.inputs))
        else:
            output_names = options.output_tensors.split(",")
        output_tensors = [tf.get_default_graph().get_tensor_by_name(name) for name in output_names]

        export_tf(sess, folder=options.output_path, inputs=input_tensors, outputs=output_tensors)
