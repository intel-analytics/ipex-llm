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

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.tf.{Const => ConstOps}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.{NumericBoolean, NumericChar, NumericDouble, NumericFloat, NumericInt, NumericLong, NumericShort, NumericString}
import com.intel.analytics.bigdl.utils.tf.TFTensorNumeric.NumericByteString
import com.intel.analytics.bigdl.utils.tf.{Context, TFUtils}
import org.tensorflow.framework.NodeDef

import scala.reflect.ClassTag

class Const extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder,
    context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    val value = TFUtils.parseTensor(nodeDef.getAttrMap.get("value").getTensor, byteOrder)
    val const = value.getTensorNumeric() match {
      case NumericFloat => ConstOps[T, Float](value.asInstanceOf[Tensor[Float]])
      case NumericDouble => ConstOps[T, Double](value.asInstanceOf[Tensor[Double]])
      case NumericInt => ConstOps[T, Int](value.asInstanceOf[Tensor[Int]])
      case NumericLong => ConstOps[T, Long](value.asInstanceOf[Tensor[Long]])
      case NumericChar => ConstOps[T, Char](value.asInstanceOf[Tensor[Char]])
      case NumericBoolean => ConstOps[T, Boolean](value.asInstanceOf[Tensor[Boolean]])
      case NumericShort => ConstOps[T, Short](value.asInstanceOf[Tensor[Short]])
      case NumericString => ConstOps[T, String](value.asInstanceOf[Tensor[String]])
      case NumericByteString => ConstOps[T, ByteString](value.asInstanceOf[Tensor[ByteString]])
    }
    const.asInstanceOf[Module[T]]
  }
}

