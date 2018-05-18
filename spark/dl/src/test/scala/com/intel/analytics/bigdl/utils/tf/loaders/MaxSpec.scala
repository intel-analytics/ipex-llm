/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.utils.tf.loaders

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.tf.Tensorflow.{typeAttr, booleanAttr}
import com.intel.analytics.bigdl.utils.tf.TensorflowSpecHelper
import org.tensorflow.framework.{DataType, NodeDef}

class MaxSpec extends TensorflowSpecHelper{
  "Max" should "be correct for float" in {
    compare[Float](
      NodeDef.newBuilder()
        .setName("max_test")
        .putAttr("keep_dims", booleanAttr(true))
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("Tidx", typeAttr(DataType.DT_INT32))
        .setOp("Max"),
      Seq(Tensor[Float].range(1, 10).resize(5, 2),
        Tensor.scalar[Int](1)),
      0
    )
  }

  "Max" should "be correct for Int" in {
    compare[Float](
      NodeDef.newBuilder()
        .setName("max_test")
        .putAttr("keep_dims", booleanAttr(true))
        .putAttr("T", typeAttr(DataType.DT_INT32))
        .putAttr("Tidx", typeAttr(DataType.DT_INT32))
        .setOp("Max"),
      Seq(Tensor[Int].range(1, 10).resize(5, 2),
        Tensor.scalar[Int](1)),
      0
    )
  }

  "Max" should "be correct for double" in {
    compare[Float](
      NodeDef.newBuilder()
        .setName("max_test")
        .putAttr("keep_dims", booleanAttr(false))
        .putAttr("T", typeAttr(DataType.DT_DOUBLE))
        .putAttr("Tidx", typeAttr(DataType.DT_INT32))
        .setOp("Max"),
      Seq(Tensor[Double].range(1, 10).resize(5, 2),
        Tensor.scalar[Int](1)),
      0
    )
  }

}
