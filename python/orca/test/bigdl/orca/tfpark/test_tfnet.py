#
# Copyright 2018 Analytics Zoo Authors.
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

import pytest


from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.tfpark import TFNet, TFDataset
from bigdl.util.common import *

np.random.seed(1337)  # for reproducibility


class TestTF(ZooTestCase):

    resource_path = os.path.join(os.path.split(__file__)[0], "../resources")

    def test_init_tf_net(self):
        tfnet_path = os.path.join(TestTF.resource_path, "tfnet")
        net = TFNet.from_export_folder(tfnet_path)
        output = net.forward(np.random.rand(2, 4))
        assert output.shape == (2, 2)

    def test_for_scalar(self):
        import tensorflow as tf
        with tf.Graph().as_default():
            input1 = tf.placeholder(dtype=tf.float32, shape=())
            output = input1 + 1
            sess = tf.Session()
            net = TFNet.from_session(sess, [input1], [output])
            sess.close()
        out_value = net.forward(np.array(1.0))
        assert len(out_value.shape) == 0

        # the following test would fail on bigdl 0.6.0 due to a bug in bigdl,
        # comment it out for now

        # out_value = net.predict(np.array([1.0])).first()
        # assert len(out_value.shape) == 0

    def test_init_tfnet_from_session(self):
        import tensorflow as tf
        with tf.Graph().as_default():
            input1 = tf.placeholder(dtype=tf.float32, shape=(None, 2))
            label1 = tf.placeholder(dtype=tf.float32, shape=(None, 1))
            hidden = tf.layers.dense(input1, 4)
            output = tf.layers.dense(hidden, 1)
            loss = tf.reduce_mean(tf.square(output - label1))
            grad_inputs = tf.gradients(loss, input1)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                data = np.random.rand(2, 2)
                output_value_ref = sess.run(output, feed_dict={input1: data})
                label_value = output_value_ref - 1.0
                grad_input_value_ref = sess.run(grad_inputs[0],
                                                feed_dict={input1: data,
                                                           label1: label_value})
                net = TFNet.from_session(sess, [input1], [output], generate_backward=True)

        output_value = net.forward(data)

        grad_input_value = net.backward(data, np.ones(shape=(2, 1)))

        self.assert_allclose(output_value, output_value_ref)
        self.assert_allclose(grad_input_value, grad_input_value_ref)

    def test_init_tfnet_from_saved_model(self):
        model_path = os.path.join(TestTF.resource_path, "saved-model-resource")
        tfnet = TFNet.from_saved_model(model_path, inputs=["flatten_input:0"],
                                       outputs=["dense_2/Softmax:0"])
        result = tfnet.predict(np.ones(dtype=np.float32, shape=(20, 28, 28, 1)))
        result.collect()

    def test_tf_net_predict(self):
        tfnet_path = os.path.join(TestTF.resource_path, "tfnet")
        import tensorflow as tf
        tf_session_config = tf.ConfigProto(inter_op_parallelism_threads=1,
                                           intra_op_parallelism_threads=1)
        net = TFNet.from_export_folder(tfnet_path, tf_session_config=tf_session_config)
        output = net.predict(np.random.rand(16, 4), batch_per_thread=5, distributed=False)
        assert output.shape == (16, 2)

    def test_tf_net_predict_dataset(self):
        tfnet_path = os.path.join(TestTF.resource_path, "tfnet")
        net = TFNet.from_export_folder(tfnet_path)
        dataset = TFDataset.from_ndarrays((np.random.rand(16, 4),))
        output = net.predict(dataset)
        output = np.stack(output.collect())
        assert output.shape == (16, 2)


if __name__ == "__main__":
    pytest.main([__file__])
