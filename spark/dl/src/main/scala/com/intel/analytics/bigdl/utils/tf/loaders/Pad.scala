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
import com.intel.analytics.bigdl.nn.{Padding, Sequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.tf.{Context, TFUtils}
import org.tensorflow.framework.NodeDef

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class Pad extends TensorflowOpsLoader {

  import Utils._

  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder,
    context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    new PadLoadTF[T]()
  }
}

class PadLoadTF[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends Adapter[T](Array(2)) {
  override def build(tensorArrays: Array[Tensor[_]]): AbstractModule[Activity, Activity, T] = {
    val paddings = tensorArrays(0).asInstanceOf[Tensor[Int]]
    val pad = ArrayBuffer[Int]()
    val padding = Sequential[T]()

    for(dim <- 1 to paddings.size(1)) {
      if (paddings.valueAt(dim, 1) != 0 || paddings.valueAt(dim, 2) != 0 ) {
        if (paddings(Array(dim, 1)) != 0) {
          padding.add(Padding[T](dim, -paddings.valueAt(dim, 1), 4))
        }
        if (paddings(Array(dim, 2)) != 0) {
          padding.add(Padding[T](dim, paddings.valueAt(dim, 2), 4))
        }
      }
    }

    padding
  }
}

