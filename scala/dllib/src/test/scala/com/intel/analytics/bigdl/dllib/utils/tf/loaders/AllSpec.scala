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
import com.intel.analytics.bigdl.utils.tf.Tensorflow.{booleanAttr, typeAttr}
import com.intel.analytics.bigdl.utils.tf.TensorflowSpecHelper
import org.tensorflow.framework.{DataType, NodeDef}

class AllSpec extends TensorflowSpecHelper {
  "All ops" should "be correct when keep_dims is true" in {
    val data = Tensor[Boolean](T(T(true, true, false), T(false, true, true)))
    val indice = Tensor[Int](T(1))
    val (t1, t2) = getResult[Float, Boolean](
      NodeDef.newBuilder()
        .setName("all_test")
        .putAttr("keep_dims", booleanAttr(true))
        .setOp("All"),
      Seq(data, indice),
      0
    )
    t1.map(t2, (a, b) => {
      require(a == b, "output not match")
      b
    })
  }

  "All ops" should "be correct when indice contains several value" in {
    val data = Tensor[Boolean](T(T(true, true, false), T(false, true, true)))
    val indice = Tensor[Int](T(0, 1))
    val (t1, t2) = getResult[Float, Boolean](
      NodeDef.newBuilder()
        .setName("all_test")
        .putAttr("keep_dims", booleanAttr(true))
        .setOp("All"),
      Seq(data, indice),
      0
    )
    t1.map(t2, (a, b) => {
      require(a == b, "output not match")
      b
    })
  }

  "All ops" should "be correct when keep_dims is false" in {
    val data = Tensor[Boolean](T(T(true, true, false), T(false, true, true)))
    val indice = Tensor[Int](T(1))
    val (t1, t2) = getResult[Float, Boolean](
      NodeDef.newBuilder()
        .setName("all_test")
        .putAttr("keep_dims", booleanAttr(false))
        .setOp("All"),
      Seq(data, indice),
      0
    )
    t1.map(t2, (a, b) => {
      require(a == b, "output not match")
      b
    })
  }

  "All ops" should "be correct when indice is scalar" in {
    val data = Tensor[Boolean](T(T(true, true, false), T(false, true, true)))
    val indice = Tensor.scalar[Int](1)
    val (t1, t2) = getResult[Float, Boolean](
      NodeDef.newBuilder()
        .setName("all_test")
        .putAttr("keep_dims", booleanAttr(false))
        .setOp("All"),
      Seq(data, indice),
      0
    )
    t1.map(t2, (a, b) => {
      require(a == b, "output not match")
      b
    })
  }
}
