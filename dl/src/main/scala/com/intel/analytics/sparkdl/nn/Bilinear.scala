/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.utils.RandomGenerator._
import com.intel.analytics.sparkdl.utils.Table

import scala.reflect.ClassTag

class Bilinear[A <: Table : ClassTag, B <: Tensor[T] : ClassTag, T: ClassTag](inputSize1: Int,
  inputSize2: Int,
  outputSize: Int,
  biasRes: Boolean = true
 )(implicit ev: TensorNumeric[T]) extends Module[A, B, T] {

  require((inputSize1 > 0) && (inputSize2 > 0) && (outputSize > 0))

  val weight = Tensor[T](outputSize, inputSize1, inputSize2)
  this.gradWeight = Tensor[T](outputSize, inputSize1, inputSize2)

  val bias: Tensor[T] = if (biasRes)Tensor[T](outputSize) else null
  this.gradBias = if (biasRes) Tensor[T](outputSize) else null

  var buff2 = Tensor[T]()
  var buff1 = Tensor[T]()

  reset()

  override def reset(): Unit = {
    val stdv = 1.0 / math.sqrt(weight.size(2))
    weight.apply1(_ => ev.fromType[Double](RNG.uniform(-stdv, stdv)))
    if (null != bias ) bias.apply1(_ => ev.fromType[Double](RNG.uniform(-stdv, stdv)))
  }

  override def updateOutput(input: A): B = {
    val result = input.asInstanceOf[Table]
    val res1 = result.apply[Tensor[T]](1)
    val res2 = result.apply[Tensor[T]](2)

    require(result.length() == 2)
    require(res1.nDimension() == 2 && res2.nDimension() == 2 && res1.size(1) == res2.size(1))
    require(res1.size(2) == weight.size(2) && res2.size(2) == weight.size(3))

    // --set up buffer
    buff2.resizeAs(res2)

    // --compute output scores
    output.resize(res1.size(1), weight.size(1))
    for(k <- 1 to weight.size(1)) {
      buff2.zero()
      buff2.addmm(res1, weight(k))
      buff2.cmul(res2)
      output.narrow(2, k, 1).sum(buff2, 2)
    }
    if (bias != null) {
      output.add(bias.reshape(Array(1, bias.nElement())).expand(output.size()))
    }
    output
  }

  override def updateGradInput(input: A, gradOutput: B): A = {
    val result = input.asInstanceOf[Table]
    val res1 = result.apply[Tensor[T]](1)
    val res2 = result.apply[Tensor[T]](2)

    require(res1.size(1) == gradOutput.size(1))
    require(gradOutput.size(2) == weight.size(1))

    val gradInput = new Table() // this.gradInput.asInstanceOf[Table]
    gradInput(1) = Tensor[T]()
    gradInput(2) = Tensor[T]()

    // compute d output / d input:
    gradInput.apply[Tensor[T]](1).resizeAs(res1).fill(ev.fromType(0))
    gradInput.apply[Tensor[T]](2).resizeAs(res2).fill(ev.fromType(0))

    // do first slice of weight tensor (k = 1)
    gradInput.apply[Tensor[T]](1).addmm(res2, weight(1).t())
    gradInput.apply[Tensor[T]](1).cmul(gradOutput.narrow(2, 1, 1).expand(
      Array(gradInput.apply[Tensor[T]](1).size(1), gradInput.apply[Tensor[T]](1).size(2))))

    gradInput.apply[Tensor[T]](2).addmm(ev.fromType(1), res1, weight(1))
    gradInput.apply[Tensor[T]](2).cmul(gradOutput.narrow(2, 1, 1).expand(
      Array(gradInput.apply[Tensor[T]](2).size(1), gradInput.apply[Tensor[T]](2).size(2))))

    // --do remaing slices of weight tensor
    if(weight.size(1) > 1) {
      buff1.resizeAs(res1)

      println(weight.size(1))
      for(k <- 2 to weight.size(1)) {
        buff1.zero()
        buff2.zero()

        buff1.addmm(res2, weight(k).t())
        buff1.cmul(gradOutput.narrow(2, k, 1).expand(
          Array(gradInput.apply[Tensor[T]](1).size(1), gradInput.apply[Tensor[T]](1).size(2))))
        gradInput.apply[Tensor[T]](1).add(buff1)

        buff2.addmm(input(1), weight(k))
        buff2.cmul(gradOutput.narrow(2, k, 1).expand(
          Array(gradInput.apply[Tensor[T]](2).size(1), gradInput.apply[Tensor[T]](2).size(2))))
        gradInput.apply[Tensor[T]](2).add(buff2)
      }
    }
    gradInput.asInstanceOf[A]
  }

  override def accGradParameters(input: A, gradOutput: B, scale: Double = 1.0): Unit = {
    val result = input.asInstanceOf[Table]
    val res1 = result.apply[Tensor[T]](1)
    val res2 = result.apply[Tensor[T]](2)

    // --make sure we have buffer
    buff1.resizeAs(res1)

    // --accumulate parameter gradients:
    for (k <- 1 to weight.size(1)) {
      buff1.zero()
      buff1.cmul(res1, gradOutput.narrow(2, k, 1).expandAs(res1))
      gradWeight(k).addmm(buff1.t(), input(2))
    }
    if(null != bias) gradBias.add(ev.fromType(scale), gradOutput.sum(1))
  }

  override def zeroGradParameters(): Unit = {
    gradWeight.zero()
    gradBias.zero()
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
  }

  override def toString(): String = {
    s"nn.Bilinear"
  }
}
