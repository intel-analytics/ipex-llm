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
package com.intel.analytics.bigdl.dllib.feature.image3d

import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.tensor.{DoubleType, FloatType, Tensor}
import scala.reflect.ClassTag


object WarpTransformer {
  def apply(flowField: Tensor[Double],
            offset: Boolean = true,
            clampMode: String = "clamp", padVal: Double = 0): WarpTransformer = {
    new WarpTransformer(flowField, offset, clampMode, padVal)
  }
}

private[bigdl] class WarpTransformer(flowField: Tensor[Double],
   offset: Boolean, clampMode: String, padVal: Double)
extends Serializable {
  private val _clampMode = clampMode match {
    case "clamp" => 1
    case "padding" => 2
  }

  def apply[@specialized(Float, Double) T: ClassTag](src: Tensor[T], dst: Tensor[T])
  (implicit ev: TensorNumeric[T]): Unit = {
    val depth = dst.size(1)
    val height = dst.size(2)
    val width = dst.size(3)
    val src_depth = src.size(1)
    val src_height = src.size(2)
    val src_width = src.size(3)
    val offset_mode = offset match {
      case true => 1
      case false => 0
    }

    for(z <- 1 to depth; y <- 1 to height; x <- 1 to width) {
      val flow_z = flowField.valueAt(1, z, y, x)
      val flow_y = flowField.valueAt(2, z, y, x)
      val flow_x = flowField.valueAt(3, z, y, x)
      var iz = offset_mode * z + flow_z
      var iy = offset_mode * y + flow_y
      var ix = offset_mode * x + flow_x

      // borders
      var off_image = 0
      if(iz < 1 || iz > src_depth ||
        iy < 1 || iy > src_height ||
        ix < 1 || ix > src_width) {
          off_image = 1
        }
        if(off_image == 1 && clampMode == 2) {
          dst.setValue(z, y, x, ev.fromType[Double](padVal))
        } else {
          iz = math.max(iz, 1);iz = math.min(iz, src_depth)
          iy = math.max(iy, 1);iy = math.min(iy, src_height)
          ix = math.max(ix, 1);ix = math.min(ix, src_width)

          val iz_0 = math.floor(iz).toInt
          val iy_0 = math.floor(iy).toInt
          val ix_0 = math.floor(ix).toInt
          val iz_1 = math.min(iz_0 + 1, src_depth)
          val iy_1 = math.min(iy_0 + 1, src_height)
          val ix_1 = math.min(ix_0 + 1, src_width)
          val wz = iz - iz_0
          val wy = iy - iy_0
          val wx = ix - ix_0
          val value =
              (1 - wy) * (1 - wx) * (1 - wz) * ev.toType[Double](src.valueAt(iz_0, iy_0, ix_0)) +
              (1 - wy) * (1 - wx) * wz * ev.toType[Double](src.valueAt(iz_1, iy_0, ix_0)) +
              (1 - wy) * wx * (1 - wz) * ev.toType[Double](src.valueAt(iz_0, iy_0, ix_1)) +
              (1 - wy) * wx * wz * ev.toType[Double](src.valueAt(iz_1, iy_0, ix_1)) +
              wy * (1 - wx) * (1 - wz) * ev.toType[Double](src.valueAt(iz_0, iy_1, ix_0)) +
              wy * (1 - wx) * wz * ev.toType[Double](src.valueAt(iz_1, iy_1, ix_0)) +
              wy * wx * (1-wz) * ev.toType[Double](src.valueAt(iz_0, iy_1, ix_1)) +
              wy * wx * wz * ev.toType[Double](src.valueAt(iz_1, iy_1, ix_1))
          dst.setValue(z, y, x, ev.fromType[Double](value))
        }
    }
  }
}

