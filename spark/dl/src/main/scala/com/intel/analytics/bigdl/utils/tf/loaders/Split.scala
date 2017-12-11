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
import com.intel.analytics.bigdl.nn.ConcatTable
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.tf.SplitAndSelect
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.tf.Context
import org.tensorflow.framework.NodeDef

import scala.reflect.ClassTag

class Split extends TensorflowOpsLoader {

  import Utils._

  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder,
    context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    val numSplit = nodeDef.getAttrMap.get("num_split").getI.toInt
    new SplitLoadTF[T](numSplit)
  }
}

class SplitLoadTF[T: ClassTag](val numSplit: Int)(implicit ev: TensorNumeric[T])
  extends Adapter[T](Array(1)) {
  override def build(tensorArrays: Array[Tensor[_]]): AbstractModule[Activity, Activity, T] = {
    val dim = tensorArrays(0).asInstanceOf[Tensor[Int]].value() + 1
    val model = new ConcatTable[T]()
    for (index <- Range(1, numSplit + 1)) {
      model.add(SplitAndSelect[T](dim, index, numSplit))
    }
    model
  }
}

