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
import com.intel.analytics.bigdl.nn.Sequential
import com.intel.analytics.bigdl.nn.tf.Mean
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.tf.Context
import org.tensorflow.framework.{DataType, NodeDef}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class Mean extends TensorflowOpsLoader {

  import Utils._

  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    val attr = nodeDef.getAttrMap
    val dataType = getType(attr, "T")
    val squeeze = !getBoolean(attr, "keep_dims")
    val dt = dataType match {
      case DataType.DT_INT8 =>
        "Int"
      case DataType.DT_INT16 =>
        "Int"
      case DataType.DT_UINT8 =>
        "Int"
      case DataType.DT_UINT16 =>
        "Int"
      case DataType.DT_INT32 =>
        "Int"
      case DataType.DT_INT64 =>
        "Long"
      case DataType.DT_FLOAT =>
        "Float"
      case DataType.DT_DOUBLE =>
        "Double"
      case _ => throw new UnsupportedOperationException("Data Type: " + dataType +
        " is not Unsupported yet.")
    }
    new MeanLoadTF[T](dt, squeeze)
  }
}

class MeanLoadTF[T: ClassTag](val dataType: String,
                              val squeeze: Boolean)(implicit ev: TensorNumeric[T])
  extends Adapter[T](Array(2)) {
  override def build(tensorArrays: Array[Tensor[_]]): AbstractModule[Activity, Activity, T] = {
    val dims = tensorArrays(0).asInstanceOf[Tensor[Int]]
    val dim = ArrayBuffer[Int]()
    val mean = Sequential[T]()
    for (i <- 1 to dims.size(1)) {
      dim += dims.valueAt(i) + 1
    }
    dataType match {
      case "Int" =>
        dim.foreach(i => mean.add(Mean[T, Int](i, squeeze = squeeze)))
      case "Long" =>
        dim.foreach(i => mean.add(Mean[T, Long](i, squeeze = squeeze)))
      case "Float" =>
        dim.foreach(i => mean.add(Mean[T, Float](i, squeeze = squeeze)))
      case "Double" =>
        dim.foreach(i => mean.add(Mean[T, Double](i, squeeze = squeeze)))
      case _ => throw new UnsupportedOperationException("Data Type: " + dataType +
        " is not Unsupported yet.")
    }
    mean
  }
}
