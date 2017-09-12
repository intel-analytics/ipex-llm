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

class RandomUniform[T: ClassTag, DataType: ClassTag](
  minVal: DataType,
  maxVal: DataType,
  seed: Int = 0
)
  (implicit ev: TensorNumeric[T], ev2: TensorNumeric[DataType])
extends Operation[Tensor[DataType], Tensor[DataType], T] {

  RandomGenerator.RNG.setSeed(seed)

  output = Activity.allocate[Tensor[DataType], DataType]()
  override def updateOutput(input: Tensor[DataType]): Tensor[DataType] = {
    require(input.nDimension() == 1, "the shape should be a one-dimensional tensor.")

    val shape = input.asInstanceOf[Tensor[Int]].storage().toArray
    output.resize(shape).rand(
      minVal.asInstanceOf[Double],
      maxVal.asInstanceOf[Double])

    output
  }
}

object RandomUniform {
  def apply[T: ClassTag, DataType: ClassTag](
    minVal: DataType,
    maxVal: DataType,
    seed: Int = 0)
    (implicit ev: TensorNumeric[T], ev2: TensorNumeric[DataType]):
  Operation[Activity, Activity, T]
  = ModuleToOperation[T](new RandomUniform(minVal, maxVal, seed))
}