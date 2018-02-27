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
package com.intel.analytics.bigdl.nn.ops

import com.intel.analytics.bigdl.nn.ResizeBilinear
import com.intel.analytics.bigdl.nn.abstractnn.{Activity, DataFormat}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class ResizeBilinearOps[T: ClassTag](alignCorner: Boolean)(implicit ev: TensorNumeric[T])
  extends Operation[Activity, Tensor[T], T] {

  private var module : ResizeBilinear[T] = _

  override def updateOutput(input: Activity): Tensor[T] = {
    require(input.isTable, "Only accept two input tensors")
    val size = input.toTable.apply[Tensor[Int]](2)
    if (module == null) {
      module = ResizeBilinear[T](
        size.valueAt(1),
        size.valueAt(2),
        alignCorner,
        dataFormat = DataFormat.NHWC
      )
    } else {
      require(module.outputHeight == size.valueAt(1), "height not match")
      require(module.outputWidth == size.valueAt(2), "width not match")
    }
    val data = input.toTable.apply[Tensor[T]](1)
    output = module.forward(data)
    output
  }
}

object ResizeBilinearOps {
  def apply[T: ClassTag](alignCorner: Boolean)
    (implicit ev: TensorNumeric[T]): ResizeBilinearOps[T] = {
    new ResizeBilinearOps(alignCorner)
  }
}

private[bigdl] class ResizeBilinearGrad[T: ClassTag](alignCorner: Boolean)
  (implicit ev: TensorNumeric[T]) extends Operation[Activity, Tensor[T], T] {

  private var module : ResizeBilinear[T] = _

  override def updateOutput(input: Activity): Tensor[T] = {
    require(input.isTable, "Only accept two input tensors")
    val grads = input.toTable.apply[Tensor[T]](1)
    val originImage = input.toTable.apply[Tensor[T]](2)
    if (module == null) {
      module = ResizeBilinear[T](
        grads.size(2),
        grads.size(3),
        alignCorner,
        dataFormat = DataFormat.NHWC
      )
    } else {
      require(module.outputHeight == grads.size(2), "height not match")
      require(module.outputWidth == grads.size(3), "width not match")
    }
    output = module.backward(originImage, grads)
    output
  }
}

private[bigdl] object ResizeBilinearGrad {
  def apply[T: ClassTag](alignCorner: Boolean)
                        (implicit ev: TensorNumeric[T]): ResizeBilinearGrad[T] = {
    new ResizeBilinearGrad[T](alignCorner)
  }
}
