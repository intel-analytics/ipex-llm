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

package com.intel.analytics.bigdl.nn.keras

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn._
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag


@SerialVersionUID( 359656776803598944L)
class Dense[T: ClassTag](val outputDim: Int,
                         val init: InitializationMethod = RandomUniform,
                         var wRegularizer: Regularizer[T] = null,
                         var bRegularizer: Regularizer[T] = null,
                         val bias: Boolean = true,
                         var inputShape: Array[Int] = null
  )(implicit ev: TensorNumeric[T]) extends NewModule[Tensor[T], Tensor[T], T](inputShape) {

  override def doBuild(inputShape: Activity): AbstractModule[Tensor[T], Tensor[T], T] = {
    val model = Linear(
      inputSize = inputShape.toTensor[Int].toArray()(1),  // zero position is batch
      outputSize = outputDim,
      withBias = bias,
      wRegularizer = wRegularizer,
      bRegularizer = bRegularizer
    )
    model.setInitMethod(weightInitMethod = init, biasInitMethod = Zeros)
    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

