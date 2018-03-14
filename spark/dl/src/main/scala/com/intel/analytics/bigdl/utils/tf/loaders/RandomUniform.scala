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
import com.intel.analytics.bigdl.nn.ops.{RandomUniform => RandomUniformOps}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.tf.Context
import org.tensorflow.framework.{DataType, NodeDef}

import scala.reflect.ClassTag

class RandomUniform extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder,
    context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    val seed = if (nodeDef.getAttrMap.containsKey("seed")) {
      Some(nodeDef.getAttrMap.get("seed").getI().toInt)
    } else {
      None
    }

    nodeDef.getAttrMap.get("dtype").getType match {
      case DataType.DT_FLOAT =>
        val min = 0
        val max = 1
        RandomUniformOps[T, Float](min, max, seed)
      case DataType.DT_DOUBLE =>
        val min = 0
        val max = 1
        RandomUniformOps[T, Double](min, max, seed)
      case _ =>
        throw new IllegalArgumentException("Not support data type")
    }
  }
}
