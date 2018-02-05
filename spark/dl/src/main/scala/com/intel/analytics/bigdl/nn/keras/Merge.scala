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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.{CAddTable, CAveTable, CMaxTable, CMulTable, CosineDistance, DotProduct, JoinTable, ParallelTable, Sequential => TSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{MultiShape, Shape}

import scala.reflect.ClassTag

class Merge[T: ClassTag](
   val layers: Array[AbstractModule[Activity, Activity, T]] = null,
   val mode: String = "sum",
   val concatAxis: Int = -1,
   var inputShape: MultiShape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  private val mergeMode = mode.toLowerCase()
  private var axis = concatAxis

  require(mergeMode == "sum" || mergeMode == "mul" || mergeMode == "concat" || mergeMode == "ave"
  || mergeMode == "cos" || mergeMode == "dot" || mergeMode == "max",
  s"Invalid merge mode: $mergeMode")
  require(layers.length >= 2, s"Merge must have at least two layers but found ${layers.length}")

  override def getInputShape(): Shape = {
    inputShape = if (inputShape != null) inputShape
    else {
      MultiShape(layers.map { layer =>
        layer.build(layer.getInputShape())
      }.toList)
    }
    inputShape
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toMulti()
    val input1 = input.head.toSingle().toArray
    if (mergeMode == "concat") {
      import scala.util.control.Breaks._
      val output = input1.clone()
      require(Math.abs(concatAxis) < output.length, s"Invalid concat axis $concatAxis")
      axis = if (concatAxis < 0) concatAxis + output.length else concatAxis
      var i = 1
      while (i < input.length) {
        val input_i = input(i).toSingle().toArray
        var j = 0
        while (j < input_i.length) {
          if (j != axis) require(input_i(j)==output(j), s"Incompatible input dimension for merge " +
            s"mode concat: (${output.deep.mkString(", ")}), " +
            s"(${input_i.deep.mkString(", ")})")
          j += 1
        }
        if (output(axis) == -1 || input_i(axis) == -1) {
          output(i) = -1
          break
        }
        output(axis) = output(axis) + input_i(axis)
        i += 1
      }
      Shape(output)
    }
    else {
      var i = 1
      while (i < input.length) {
        val input_i = input(i).toSingle().toArray
        require(input_i.sameElements(input1), s"Incompatible input dimension for " +
          s"merge mode $mergeMode: (${input1.deep.mkString(", ")}), " +
          s"(${input_i.deep.mkString(", ")})")
        i += 1
      }
      if (mergeMode == "dot" || mergeMode == "cos") {
        require(input.head.toSingle().length <=2, s"For merge mode $mergeMode, 3D input " +
          s"or above is not supported, got input dim ${input.head.toSingle().length}")
        require(input.length == 2, s"Merge mode $mergeMode takes exactly two layers")
        if (mergeMode == "dot") Shape(-1, 1) else Shape(-1, 1, 1)
      }
      else {
        input.head
      }
    }
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toMulti()
    val model = TSequential[T]()
    val parallel = ParallelTable()
    var i = 0
    while(i < layers.length) {
      val tlayer = layers(i) match {
        case k: KerasLayer[_, _, T] => k.labor
        case t: AbstractModule[Activity, Activity, T] => t
      }
      parallel.add(tlayer)
      i += 1
    }
    model.add(parallel)
    val seq = TSequential[T]()
    val layer =
      if (mergeMode == "sum") CAddTable()
      else if (mergeMode == "mul") CMulTable()
      else if (mergeMode == "max") CMaxTable()
      else if (mergeMode == "ave") CAveTable()
      else if (mergeMode == "concat") JoinTable(axis, input.length)
      else if (mergeMode == "dot") {
        require(input.head.toSingle().length <=2, s"For merge mode dot, only 1D or 2D " +
          s"input is supported, but got input dim ${input.head.toSingle().length}")
        seq.add(DotProduct())
        seq.add(com.intel.analytics.bigdl.nn.Reshape(Array(1), Some(true)))
        seq
      }
      else {
        seq.add(CosineDistance())
        seq.add(com.intel.analytics.bigdl.nn.Reshape(Array(1, 1), Some(true)))
        seq
      }
    model.add(layer)
    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object Merge {
  def apply[@specialized(Float, Double) T: ClassTag](
    layers: List[AbstractModule[Activity, Activity, T]] = null,
    mode: String = "sum",
    concatAxis: Int = -1,
    inputShape: MultiShape = null)(implicit ev: TensorNumeric[T]): Merge[T] = {
    new Merge[T](layers.toArray, mode, concatAxis, inputShape)
  }
}
