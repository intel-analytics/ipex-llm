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
import com.intel.analytics.bigdl.utils.tf.Tensorflow.intAttr
import com.intel.analytics.bigdl.utils.tf.TensorflowSpecHelper
import org.tensorflow.framework.NodeDef

class ArrayOps extends TensorflowSpecHelper {
  "InvertPermutation" should "be correct" in {
    compare[Float](
      NodeDef.newBuilder()
        .setName("invert_permutation_test")
        .setOp("InvertPermutation"),
      Seq(Tensor[Int](T(3, 4, 0, 2, 1))),
      0
    )
  }

  "ConcatOffset" should "be correct" in {
    compare[Float](
      NodeDef.newBuilder()
        .setName("concat_offset_test")
        .putAttr("N", intAttr(3))
        .setOp("ConcatOffset"),
      Seq(Tensor.scalar[Int](1), Tensor[Int](T(2, 2, 5, 7)), Tensor[Int](T(2, 3, 5, 7)),
        Tensor[Int](T(2, 4, 5, 7))),
      Seq(0, 1, 2), 1e-5
    )
  }
}
