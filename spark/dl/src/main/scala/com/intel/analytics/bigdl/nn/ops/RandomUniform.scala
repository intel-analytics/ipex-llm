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

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.RandomGenerator

import scala.reflect.ClassTag

private[bigdl] trait RandomNode

class RandomUniform[T: ClassTag, D: ClassTag](
  minVal: D, maxVal: D, seed: Option[Int] = None
)(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Tensor[Int], Tensor[D], T] with RandomNode {

  if (seed.isDefined) {
    RandomGenerator.RNG.setSeed(seed.get)
  }

  output = Activity.allocate[Tensor[D], D]()

  override def updateOutput(input: Tensor[Int]): Tensor[D] = {
    require(input.nDimension() == 1, "the shape should be a one-dimensional tensor.")

    val shape = input.storage().toArray
    output.resize(shape).rand(
      ev2.toType[Double](minVal),
      ev2.toType[Double](maxVal))

    output
  }
}

object RandomUniform {
  def apply[T: ClassTag, D: ClassTag](
    minVal: D,
    maxVal: D,
    seed: Option[Int] = None)
    (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]):
  Operation[Activity, Activity, T]
  = ModuleToOperation[T](new RandomUniform(minVal, maxVal, seed))
}
