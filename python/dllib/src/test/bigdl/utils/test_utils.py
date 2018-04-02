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
import numpy as np
np.random.seed(1337)  # for reproducibility


class ZooTestCase(TestCase):

    def setup_method(self, method):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        keras.backend.set_image_dim_ordering("th")
        sparkConf = create_spark_conf().setMaster("local[4]").setAppName("test model")
        self.sc = get_spark_context(sparkConf)
        self.sqlContext = SQLContext(self.sc)
        init_engine()

    def teardown_method(self, method):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        keras.backend.set_image_dim_ordering("th")
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


    def compare_loss(self, y_a, y_b, kloss, bloss, rtol=1e-6, atol=1e-6):
        # y_a: input/y_pred; y_b: target/y_true
        keras_output = np.mean(K.eval(kloss(K.variable(y_b), K.variable(y_a))))
        bigdl_output = bloss.forward(y_a, y_b)
        np.testing.assert_allclose(bigdl_output, keras_output, rtol=rtol, atol=atol)

    # Compare forward results with Keras for new Keras-like API layers.
    def compare_newapi(self, klayer, blayer, input_data, weight_converter=None,
                       is_training=False, rtol=1e-6, atol=1e-6):
        from keras.models import Sequential as KSequential
        from bigdl.nn.keras.topology import Sequential as BSequential
        bmodel = BSequential()
        bmodel.add(blayer)
        kmodel = KSequential()
        kmodel.add(klayer)
        koutput = kmodel.predict(input_data)
        from bigdl.nn.keras.layer import BatchNormalization
        if isinstance(blayer, BatchNormalization):
            k_running_mean = K.eval(klayer.running_mean)
            k_running_std = K.eval(klayer.running_std)
            blayer.set_running_mean(k_running_mean)
            blayer.set_running_std(k_running_std)
        if kmodel.get_weights():
            bmodel.set_weights(weight_converter(klayer, kmodel.get_weights()))
        bmodel.training(is_training)
        boutput = bmodel.forward(input_data)
        self.assert_allclose(boutput, koutput, rtol=rtol, atol=atol)

    def compare_model(self, bmodel, kmodel, input_data, rtol=1e-5, atol=1e-5):
        WeightLoader.load_weights_from_kmodel(bmodel, kmodel)
        bmodel.training(is_training=False)
        bigdl_output = bmodel.forward(input_data)
        keras_output = kmodel.predict(input_data)
        self.assert_allclose(bigdl_output, keras_output, rtol=rtol, atol=atol)
