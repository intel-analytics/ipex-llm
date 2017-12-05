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
import com.intel.analytics.bigdl.utils.tf.TensorflowSpecHelper
import org.tensorflow.framework.{DataType, NodeDef}
import com.intel.analytics.bigdl.utils.tf.Tensorflow._

class RangeSpec extends TensorflowSpecHelper {
  "Range" should "be correct when input is int" in {
    compare[Float](
      NodeDef.newBuilder()
      .setName("range_test")
      .setOp("Range")
      .putAttr("Tidx", typeAttr(DataType.DT_INT32)),
      Seq(Tensor.scalar[Int](3), Tensor.scalar[Int](18), Tensor.scalar[Int](3)), 0
    )
  }

  "Range" should "be correct when input is float" in {
    compare[Float](
      NodeDef.newBuilder()
        .setName("range_test")
        .setOp("Range")
        .putAttr("Tidx", typeAttr(DataType.DT_FLOAT)),
      Seq(Tensor.scalar[Float](3), Tensor.scalar[Float](1), Tensor.scalar[Float](-0.5f)), 0
    )
  }
}
