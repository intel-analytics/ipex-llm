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

from __future__ import print_function

import logging
import shutil
from unittest import TestCase

from bigdl.dllib.nncontext import *
from bigdl.dllib.feature.image import ImageSet
from bigdl.dllib.utils.log4Error import *


np.random.seed(1337)  # for reproducibility


class ZooTestCase(TestCase):
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('py4j').setLevel(logging.INFO)

    def setup_method(self, method):
        """
        Setup any state tied to the execution of the given method in a class.
        It is invoked for every test method of a class.
        """
        sparkConf = init_spark_conf().setMaster("local[4]").setAppName("zoo test case")\
            .set("spark.driver.memory", "5g")
        invalidInputError(str(sparkConf.get("spark.shuffle.reduceLocality.enabled")) == "false",
                          "expect spark.shuffle.reduceLocality.enabled == false in spark conf")
        invalidInputError(
            str(sparkConf.get("spark.serializer")) == "org.apache.spark.serializer.JavaSerializer",
            "expect spark.serializer == org.apache.spark.serializer.JavaSerializer in spark conf")
        invalidInputError(SparkContext._active_spark_context is None,
                          "SparkContext._active_spark_context should be none")
        self.sc = init_nncontext(sparkConf)
        self.sc.setLogLevel("ERROR")
        self.sqlContext = SQLContext(self.sc)
        self.tmp_dirs = []

    def teardown_method(self, method):
        """
        Teardown any state that was previously setup with a setup_method call.
        """
        self.sc.stop()
        if hasattr(self, "tmp_dirs"):
            for d in self.tmp_dirs:
                shutil.rmtree(d)

    def create_temp_dir(self):
        tmp_dir = tempfile.mkdtemp()
        self.tmp_dirs.append(tmp_dir)
        return tmp_dir

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

    def assert_list_allclose(self, a, b, rtol=1e-6, atol=1e-6, msg=None):
        for (i1, i2) in zip(a, b):
            self.assert_allclose(i1, i2, rtol, atol, msg)

    def assert_tfpark_model_save_load(self, model, input_data, rtol=1e-6, atol=1e-6):
        model_class = model.__class__
        tmp_path = create_tmp_path() + ".h5"
        model.save_model(tmp_path)
        loaded_model = model_class.load_model(tmp_path)
        invalidInputError(isinstance(loaded_model, model_class),
                          "loaded_model should be model_class")
        # Calling predict will remove the impact of dropout.
        output1 = model.predict(input_data)
        output2 = loaded_model.predict(input_data, distributed=True)
        if isinstance(output1, list):
            self.assert_list_allclose(output1, output2, rtol, atol)
        else:
            self.assert_allclose(output1, output2, rtol, atol)
        os.remove(tmp_path)

    def intercept(self, func, error_message):

        error = False
        try:
            func()
        except Exception as e:
            if error_message not in str(e):
                invalidInputError(False,
                                  "error_message not in the exception raised. " +
                                  "error_message: %s, exception: %s" % (error_message, e))
            error = True

        if not error:
            invalidInputError(False, "exception is not throw")
