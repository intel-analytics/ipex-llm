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

from __future__ import print_function

from unittest import TestCase
from bigdl.keras.converter import WeightLoader
import keras.backend as K
from bigdl.util.common import *

np.random.seed(1337)  # for reproducibility


class ZooTestCase(TestCase):

    def setup_method(self, method):
        """
        Setup any state tied to the execution of the given method in a class.
        It is invoked for every test method of a class.
        """
        K.set_image_dim_ordering("th")
        sparkConf = create_spark_conf().setMaster("local[4]").setAppName("zoo test case")
        self.sc = get_spark_context(sparkConf)
        self.sqlContext = SQLContext(self.sc)
        init_engine()

    def teardown_method(self, method):
        """
        Teardown any state that was previously setup with a setup_method call.
        """
        K.set_image_dim_ordering("th")
        self.sc.stop()

    def assert_allclose(self, a, b, rtol=1e-6, atol=1e-6, msg=None):
        # from tensorflow
        self.assertEqual(a.shape, b.shape, "Shape mismatch: expected %s, got %s." %
                         (a.shape, b.shape))
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            cond = np.logical_or(
                np.abs(a - b) > atol + rtol * np.abs(b), np.isnan(a) != np.isnan(b))
            if a.ndim:
                x = a[np.where(cond)]
                y = b[np.where(cond)]
                print("not close where = ", np.where(cond))
            else:
                # np.where is broken for scalars
                x, y = a, b
            print("not close lhs = ", x)
            print("not close rhs = ", y)
            print("not close dif = ", np.abs(x - y))
            print("not close tol = ", atol + rtol * np.abs(y))
            print("dtype = %s, shape = %s" % (a.dtype, a.shape))
            np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=msg)

    def compare_loss(self, y_a, y_b, kloss, zloss, rtol=1e-6, atol=1e-6):
        """
        Compare forward results for Keras loss against Zoo loss.

        # Arguments
        y_a: input/y_pred
        y_b: target/y_true
        """
        keras_output = np.mean(K.eval(kloss(K.variable(y_b), K.variable(y_a))))
        zoo_output = zloss.forward(y_a, y_b)
        np.testing.assert_allclose(zoo_output, keras_output, rtol=rtol, atol=atol)

    def compare_layer(self, klayer, zlayer, input_data, weight_converter=None,
                      is_training=False, rtol=1e-6, atol=1e-6):
        """
        Compare forward results for Keras layer against Zoo Keras API layer.
        """
        from keras.models import Sequential as KSequential
        from zoo.pipeline.api.keras.models import Sequential as ZSequential
        zmodel = ZSequential()
        zmodel.add(zlayer)
        kmodel = KSequential()
        kmodel.add(klayer)
        koutput = kmodel.predict(input_data)
        from zoo.pipeline.api.keras.layers import BatchNormalization
        if isinstance(zlayer, BatchNormalization):
            k_running_mean = K.eval(klayer.running_mean)
            k_running_std = K.eval(klayer.running_std)
            zlayer.set_running_mean(k_running_mean)
            zlayer.set_running_std(k_running_std)
        if kmodel.get_weights():
            zmodel.set_weights(weight_converter(klayer, kmodel.get_weights()))
        zmodel.training(is_training)
        zoutput = zmodel.forward(input_data)
        self.assert_allclose(zoutput, koutput, rtol=rtol, atol=atol)

    def compare_model(self, bmodel, kmodel, input_data, rtol=1e-5, atol=1e-5):
        WeightLoader.load_weights_from_kmodel(bmodel, kmodel)
        bmodel.training(is_training=False)
        bigdl_output = bmodel.forward(input_data)
        keras_output = kmodel.predict(input_data)
        self.assert_allclose(bigdl_output, keras_output, rtol=rtol, atol=atol)
