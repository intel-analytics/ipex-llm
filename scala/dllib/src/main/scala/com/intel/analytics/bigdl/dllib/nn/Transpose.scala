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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.{NumericWildcard, TensorNumeric}
import com.intel.analytics.bigdl.utils.serializer._
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag
import scala.reflect.runtime.universe

/**
 * Transpose input along specified dimensions
 * @param permutations dimension pairs that need to swap
 */
@SerialVersionUID(8543726779794064339L)
class Transpose[T: ClassTag](
  val permutations: Array[(Int, Int)])(implicit ev: TensorNumeric[T])
  extends AbstractModule[Tensor[_], Tensor[_], T] {

  var buffer: Tensor[_] = _

  override def updateOutput(input: Tensor[_]): Tensor[_] = {
    if (output.getType() != input.getType()) {
      output = input.emptyInstance()
    }
    var i = 0
    buffer = input
    while (i < permutations.length) {
      buffer = buffer.transpose(permutations(i)._1, permutations(i)._2)
      i += 1
    }
    output.resizeAs(buffer).asInstanceOf[Tensor[NumericWildcard]]
      .copy(buffer.asInstanceOf[Tensor[NumericWildcard]])
    output
  }

  override def updateGradInput(input: Tensor[_], gradOutput: Tensor[_]): Tensor[_] = {
    if (gradInput.getType() != input.getType()) {
      gradInput = input.emptyInstance()
    }
    var i = permutations.length - 1
    buffer = gradOutput
    while (i >= 0) {
      buffer = buffer.transpose(permutations(i)._1, permutations(i)._2)
      i -= 1
    }
    gradInput.resizeAs(buffer).asInstanceOf[Tensor[NumericWildcard]]
      .copy(buffer.asInstanceOf[Tensor[NumericWildcard]])
    gradInput
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    val inputSize = inputShape.toSingle().toArray
    var i = 0
    while (i < permutations.length) {
      val tmp = inputSize(permutations(i)._1 - 1)
      inputSize(permutations(i)._1 - 1) = inputSize(permutations(i)._2 - 1)
      inputSize(permutations(i)._2 - 1) = tmp
      i += 1
    }
    Shape(inputSize)
  }

  override def toString(): String = {
    s"${getPrintName}(${
      permutations.map {
        case (from: Int, to: Int) => s"$from -> $to"
      }.mkString(", ")
    })"
  }
}

object Transpose extends ModuleSerializable {
  def apply[@specialized(Float, Double) T: ClassTag](
      permutations: Array[(Int, Int)])(implicit ev: TensorNumeric[T]) : Transpose[T] = {
    new Transpose[T](permutations)
  }

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {

    val attrMap = context.bigdlModule.getAttrMap

    val size = DataConverter.
      getAttributeValue(context, attrMap.get("size")).
      asInstanceOf[Int]

    val permutations = new Array[(Int, Int)](size)

    var i = 0

    while (i < size) {
      val permutation = DataConverter.
        getAttributeValue(context, attrMap.get(s"permutation_$i")).
        asInstanceOf[Array[Int]]
      permutations(i) = (permutation(0), permutation(1))
      i += 1
    }

    Transpose(permutations).asInstanceOf[AbstractModule[Activity,
      Activity, T]]

  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                              transposeBuilder : BigDLModule.Builder)
                                           (implicit ev: TensorNumeric[T]) : Unit = {
    val transpose = context.moduleData.module.
      asInstanceOf[Transpose[T]]

    val size = transpose.permutations.length

    val sizeBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, sizeBuilder, size, universe.typeOf[Int])
    transposeBuilder.putAttr("size", sizeBuilder.build)

    var i = 0

    while (i < size) {
      val nextPermutationBuilder = AttrValue.newBuilder
      val arr : Array[Int] = Array(transpose.permutations(i)._1,
        transpose.permutations(i)._2)
      DataConverter.setAttributeValue(context, nextPermutationBuilder,
        arr, universe.typeOf[Array[Int]])
      transposeBuilder.putAttr(s"permutation_$i", nextPermutationBuilder.build)
      i += 1
    }

  }
}
