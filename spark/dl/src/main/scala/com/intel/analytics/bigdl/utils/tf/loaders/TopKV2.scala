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

import java.nio.ByteOrder

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.ops.{TopK => TopKOps}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.tf.Context
import org.tensorflow.framework.{DataType, NodeDef}

import scala.reflect.ClassTag

class TopKV2 extends TensorflowOpsLoader {

  import Utils._

  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder, context: Context[T])
    (implicit ev: TensorNumeric[T]): Module[T] = {

    val s = if (nodeDef.getAttrMap.containsKey("sorted")) {
      getBoolean(nodeDef.getAttrMap, "sorted")
    } else {
      true
    }
    val t = getType(nodeDef.getAttrMap, "T")
    val ts = if (t == DataType.DT_FLOAT) {
      "Float"
    } else if (t == DataType.DT_DOUBLE) {
      "Double"
    } else {
      throw new UnsupportedOperationException(s"Not support load Inv when type is ${t}")
    }

    new TopKV2LoadTF[T](s, ts)
  }
}

class TopKV2LoadTF[T: ClassTag](s: Boolean, t: String)(implicit ev: TensorNumeric[T])
  extends Adapter[T](Array(2)) {
  override def build(tensorArrays: Array[Tensor[_]]): AbstractModule[Activity, Activity, T] = {
    val kTensor = tensorArrays(0).asInstanceOf[Tensor[Int]]
    require(kTensor.isScalar, "Invalid input k")
    val k = kTensor.value()

    if (t == "Float") {
      TopKOps[T, Float](k, s, startIndex = 0)
    } else if (t == "Double") {
      TopKOps[T, Double](k, s, startIndex = 0)
    } else {
      throw new UnsupportedOperationException(s"Not support load Inv when type is ${t}")
    }
  }
}
