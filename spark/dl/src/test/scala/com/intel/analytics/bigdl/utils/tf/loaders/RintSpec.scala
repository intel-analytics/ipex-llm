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
import com.intel.analytics.bigdl.utils.tf.Tensorflow.typeAttr
import com.intel.analytics.bigdl.utils.tf.TensorflowSpecHelper
import org.tensorflow.framework.{DataType, NodeDef}

class RintSpec extends TensorflowSpecHelper {
  "Rint" should "be correct" in {
    compare[Float](
      NodeDef.newBuilder()
        .setName("rint_test")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .setOp("Rint"),
      Seq(Tensor[Float](T(-1.5, 0.5000001, -1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0))),
      0
    )
  }
}
