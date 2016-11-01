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
<<<<<<< HEAD
<<<<<<< HEAD
import com.intel.analytics.sparkdl.utils.{T, Table}

import scala.reflect.ClassTag

class Bilinear[T: ClassTag](inputSize1: Int,
  inputSize2: Int,
  outputSize: Int,
  biasRes: Boolean = true
 )(implicit ev: TensorNumeric[T]) extends Module[Table, Tensor[T], T] {

  require((inputSize1 > 0) && (inputSize2 > 0) && (outputSize > 0),
    "inputSize1 and inputSize2 and outputSize should be positive integer numbers")
=======
import com.intel.analytics.sparkdl.utils.Table
=======
import com.intel.analytics.sparkdl.utils.{T, Table}
>>>>>>> some modify of Bilinear

import scala.reflect.ClassTag

class Bilinear[T: ClassTag](inputSize1: Int,
  inputSize2: Int,
  outputSize: Int,
  biasRes: Boolean = true
 )(implicit ev: TensorNumeric[T]) extends Module[Table, Tensor[T], T] {

<<<<<<< HEAD
  require((inputSize1 > 0) && (inputSize2 > 0) && (outputSize > 0))
>>>>>>> add Bilinear layer and convert java.map to scala.map
=======
  require((inputSize1 > 0) && (inputSize2 > 0) && (outputSize > 0),
    "inputSize1 and inputSize2 and outputSize should be positive integer numbers")
>>>>>>> some modify of Bilinear

  val weight = Tensor[T](outputSize, inputSize1, inputSize2)
  this.gradWeight = Tensor[T](outputSize, inputSize1, inputSize2)

  val bias: Tensor[T] = if (biasRes)Tensor[T](outputSize) else null
  this.gradBias = if (biasRes) Tensor[T](outputSize) else null

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> some modify of Bilinear
  @transient
  var buff2: Tensor[T] = null
  @transient
  var buff1: Tensor[T] = null

  this.gradInput = T()
<<<<<<< HEAD
=======
  var buff2 = Tensor[T]()
  var buff1 = Tensor[T]()
>>>>>>> add Bilinear layer and convert java.map to scala.map
=======
>>>>>>> some modify of Bilinear

  reset()

  override def reset(): Unit = {
    val stdv = 1.0 / math.sqrt(weight.size(2))
    weight.apply1(_ => ev.fromType[Double](RNG.uniform(-stdv, stdv)))
    if (null != bias ) bias.apply1(_ => ev.fromType[Double](RNG.uniform(-stdv, stdv)))
  }

<<<<<<< HEAD
<<<<<<< HEAD
  override def updateOutput(input: Table): Tensor[T] = {
=======
  override def updateOutput(input: A): B = {
>>>>>>> add Bilinear layer and convert java.map to scala.map
=======
  override def updateOutput(input: Table): Tensor[T] = {
>>>>>>> some modify of Bilinear
    val result = input.asInstanceOf[Table]
    val res1 = result.apply[Tensor[T]](1)
    val res2 = result.apply[Tensor[T]](2)

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> some modify of Bilinear
    require(result.length() == 2,
      "input should be a table containing two data Tensors")
    require(res1.nDimension() == 2 && res2.nDimension() == 2 && res1.size(1) == res2.size(1),
      "input Tensors should be two-dimensional and have the same number of rows")
    require(res1.size(2) == weight.size(2) && res2.size(2) == weight.size(3),
      "dimensionality of first input and second input is erroneous")
<<<<<<< HEAD

    // --set up buffer
    if(null == buff2) buff2 = Tensor[T]()
=======
    require(result.length() == 2)
    require(res1.nDimension() == 2 && res2.nDimension() == 2 && res1.size(1) == res2.size(1))
    require(res1.size(2) == weight.size(2) && res2.size(2) == weight.size(3))

    // --set up buffer
>>>>>>> add Bilinear layer and convert java.map to scala.map
=======

    // --set up buffer
    if(null == buff2) buff2 = Tensor[T]()
>>>>>>> some modify of Bilinear
    buff2.resizeAs(res2)

    // --compute output scores
    output.resize(res1.size(1), weight.size(1))
<<<<<<< HEAD
<<<<<<< HEAD
    var k = 1
    while(k < (weight.size(1) + 1)) {
=======
    for(k <- 1 to weight.size(1)) {
>>>>>>> add Bilinear layer and convert java.map to scala.map
=======
    var k = 1
    while(k < (weight.size(1) + 1)) {
>>>>>>> some modify of Bilinear
      buff2.zero()
      buff2.addmm(res1, weight(k))
      buff2.cmul(res2)
      output.narrow(2, k, 1).sum(buff2, 2)
<<<<<<< HEAD
<<<<<<< HEAD
      k += 1
=======
>>>>>>> add Bilinear layer and convert java.map to scala.map
=======
      k += 1
>>>>>>> some modify of Bilinear
    }
    if (bias != null) {
      output.add(bias.reshape(Array(1, bias.nElement())).expand(output.size()))
    }
    output
  }

<<<<<<< HEAD
<<<<<<< HEAD
  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    val res1 = input.apply[Tensor[T]](1)
    val res2 = input.apply[Tensor[T]](2)

    require(res1.size(1) == gradOutput.size(1),
      "number of rows in gradOutput does not match input")
    require(gradOutput.size(2) == weight.size(1),
      "number of columns in gradOutput does not output size of layer")

    gradInput.insert(1, Tensor[T]())
    gradInput.insert(2, Tensor[T]())
=======
  override def updateGradInput(input: A, gradOutput: B): A = {
    val result = input.asInstanceOf[Table]
    val res1 = result.apply[Tensor[T]](1)
    val res2 = result.apply[Tensor[T]](2)
=======
  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    val res1 = input.apply[Tensor[T]](1)
    val res2 = input.apply[Tensor[T]](2)
>>>>>>> some modify of Bilinear

    require(res1.size(1) == gradOutput.size(1),
      "number of rows in gradOutput does not match input")
    require(gradOutput.size(2) == weight.size(1),
      "number of columns in gradOutput does not output size of layer")

<<<<<<< HEAD
    val gradInput = new Table() // this.gradInput.asInstanceOf[Table]
    gradInput(1) = Tensor[T]()
    gradInput(2) = Tensor[T]()
>>>>>>> add Bilinear layer and convert java.map to scala.map
=======
    gradInput.insert(1, Tensor[T]())
    gradInput.insert(2, Tensor[T]())
>>>>>>> some modify of Bilinear

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
<<<<<<< HEAD
<<<<<<< HEAD
      if (null == buff1) buff1 = Tensor[T]()
      buff1.resizeAs(res1)

      var k = 2
      while(k < (weight.size(1) + 1)) {
=======
      buff1.resizeAs(res1)

      println(weight.size(1))
      for(k <- 2 to weight.size(1)) {
>>>>>>> add Bilinear layer and convert java.map to scala.map
=======
      if (null == buff1) buff1 = Tensor[T]()
      buff1.resizeAs(res1)

      var k = 2
      while(k < (weight.size(1) + 1)) {
>>>>>>> some modify of Bilinear
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
<<<<<<< HEAD
<<<<<<< HEAD
        k += 1
      }
    }
    gradInput
  }

  override def accGradParameters(input: Table, gradOutput: Tensor[T], scale: Double = 1.0): Unit = {
=======
=======
        k += 1
>>>>>>> some modify of Bilinear
      }
    }
    gradInput
  }

<<<<<<< HEAD
  override def accGradParameters(input: A, gradOutput: B, scale: Double = 1.0): Unit = {
>>>>>>> add Bilinear layer and convert java.map to scala.map
=======
  override def accGradParameters(input: Table, gradOutput: Tensor[T], scale: Double = 1.0): Unit = {
>>>>>>> some modify of Bilinear
    val result = input.asInstanceOf[Table]
    val res1 = result.apply[Tensor[T]](1)
    val res2 = result.apply[Tensor[T]](2)

    // --make sure we have buffer
<<<<<<< HEAD
<<<<<<< HEAD
    if(null == buff1) buff1 = Tensor[T]()
    buff1.resizeAs(res1)

    // --accumulate parameter gradients:
    var k = 1
    while(k < (weight.size(1) + 1)) {
      buff1.zero()
      buff1.cmul(res1, gradOutput.narrow(2, k, 1).expandAs(res1))
      gradWeight(k).addmm(buff1.t(), input(2))
      k += 1
=======
=======
    if(null == buff1) buff1 = Tensor[T]()
>>>>>>> some modify of Bilinear
    buff1.resizeAs(res1)

    // --accumulate parameter gradients:
    var k = 1
    while(k < (weight.size(1) + 1)) {
      buff1.zero()
      buff1.cmul(res1, gradOutput.narrow(2, k, 1).expandAs(res1))
      gradWeight(k).addmm(buff1.t(), input(2))
<<<<<<< HEAD
>>>>>>> add Bilinear layer and convert java.map to scala.map
=======
      k += 1
>>>>>>> some modify of Bilinear
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
<<<<<<< HEAD
<<<<<<< HEAD
    s"nn.Bilinear($inputSize1, $inputSize2, $outputSize, $biasRes)"
=======
    s"nn.Bilinear"
>>>>>>> add Bilinear layer and convert java.map to scala.map
=======
    s"nn.Bilinear($inputSize1, $inputSize2, $outputSize, $biasRes)"
>>>>>>> some modify of Bilinear
  }
}
