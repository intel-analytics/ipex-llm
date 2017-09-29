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
import com.intel.analytics.bigdl.nn.SpatialAveragePooling
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.tf.Context
import org.tensorflow.framework.NodeDef

import scala.reflect.ClassTag

class AvgPool extends TensorflowOpsLoader {

  import Utils._

  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder,
    context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {

    val attributes = nodeDef.getAttrMap
    val format = getString(attributes, "data_format")
    val strideList = getIntList(attributes, "strides")
    val kernelList = getIntList(attributes, "ksize")

    val (strideH, strideW, ksizeH, ksizeW) = format match {
      case "NHWC" =>
        require(strideList(3) == 1, s"not support strides on depth")
        (strideList(1), strideList(2), kernelList(1), kernelList(2))
      case "NCHW" =>
        require(strideList(1) == 1, s"not support strides on depth")
        (strideList(2), strideList(3), kernelList(2), kernelList(3))
      case _ =>
        throw new IllegalArgumentException(s"not supported data format: $format")
    }

    val (pW, pH) =
      if (getString(attributes, "padding") == "SAME") {
        (-1, -1)
      } else {
        (0, 0)
      }

    SpatialAveragePooling[T](ksizeW, ksizeH, strideW, strideH, pW, pH,
      countIncludePad = false, format = DataFormat(format))
  }
}

