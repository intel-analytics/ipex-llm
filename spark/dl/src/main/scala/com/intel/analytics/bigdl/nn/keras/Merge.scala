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

/**
 * Used to merge a list of tensors into a single tensor, following some merge mode.
 * Merge must have at least two input layers.
 *
 * When using this layer as the first layer in a model, you need to provide the argument
 * inputShape for input layers (each as a Single Shape, does not include the batch dimension).
 *
 * @param layers A list of layer instances. Must be more than one layer.
 * @param mode Merge mode. String, must be one of: 'sum', 'mul', 'concat', 'ave', 'cos',
 *             'dot', 'max'. Default is 'sum'.
 * @param concatAxis Integer, axis to use in mode concat. Only specify this when mode is 'concat'.
 *                   Default is -1, meaning the last axis of the input.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class Merge[T: ClassTag](
   val layers: Array[AbstractModule[Activity, Activity, T]] = null,
   val mode: String = "sum",
   val concatAxis: Int = -1,
   // MultiShape isn't directly supported for serialization. Use Shape instead.
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](Merge.calcBatchInputShape(inputShape, layers)) {

  private val mergeMode = mode.toLowerCase()
  private var axis = concatAxis

  require(mergeMode == "sum" || mergeMode == "mul" || mergeMode == "concat" || mergeMode == "ave"
  || mergeMode == "cos" || mergeMode == "dot" || mergeMode == "max",
  s"Invalid merge mode: $mergeMode")
  require(layers.length >= 2, s"Merge must have at least two input layers " +
    s"but found ${layers.length}")

  private def computeOutputShapeForConcat(input: List[Shape]): Shape = {
    import scala.util.control.Breaks._
    val input1 = input.head.toSingle().toArray
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

  private def checkSameInputShape(input: List[Shape]): Unit = {
    val input1 = input.head.toSingle().toArray
    var i = 1
    while (i < input.length) {
      val input_i = input(i).toSingle().toArray
      require(input_i.sameElements(input1), s"Incompatible input dimension for " +
        s"merge mode $mergeMode: (${input1.deep.mkString(", ")}), " +
        s"(${input_i.deep.mkString(", ")})")
      i += 1
    }
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toMulti()
    val input1 = input.head.toSingle().toArray
    if (mergeMode == "concat") {
      computeOutputShapeForConcat(input)
    }
    else {
      checkSameInputShape(input)
      if (mergeMode == "dot" || mergeMode == "cos") {
        require(input.head.toSingle().length <=2, s"For merge mode $mergeMode, 3D input " +
          s"or above is currently not supported, got input dim ${input.head.toSingle().length}")
        require(input.length == 2, s"Merge mode $mergeMode takes exactly two layers, " +
          s"but got ${input.length}")
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
    val layer = mergeMode match {
      case "sum" => CAddTable()
      case "mul" => CMulTable()
      case "max" => CMaxTable()
      case "ave" => CAveTable()
      case "concat" => JoinTable(axis, input.length)
      case "dot" =>
        seq.add(DotProduct())
        seq.add(com.intel.analytics.bigdl.nn.Reshape(Array(1), Some(true)))
        seq
      case "cos" =>
        seq.add(CosineDistance())
        seq.add(com.intel.analytics.bigdl.nn.Reshape(Array(1, 1), Some(true)))
        seq
    }
    model.add(layer)
    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object Merge {
  def calcBatchInputShape[T: ClassTag](
    inputShape: Shape = null,
    layers: Array[AbstractModule[Activity, Activity, T]]): Shape = {
    val batchInputShape = KerasLayer.addBatch(inputShape)
    val actualInputShape =
      MultiShape(layers.map { layer =>
      layer.build(layer.getInputShape())
    }.toList)
    if (batchInputShape != null) {
      require(batchInputShape.isInstanceOf[MultiShape],
        "Merge requires inputShape to be MultiShape")
      require(batchInputShape.toMulti().equals(actualInputShape.toMulti()),
        "Actual layer input shapes are not the same as expected layer input shapes")
    }
    actualInputShape
  }

  def apply[@specialized(Float, Double) T: ClassTag](
    layers: List[AbstractModule[Activity, Activity, T]] = null,
    mode: String = "sum",
    concatAxis: Int = -1,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Merge[T] = {
    new Merge[T](layers.toArray, mode, concatAxis, inputShape)
  }
}
