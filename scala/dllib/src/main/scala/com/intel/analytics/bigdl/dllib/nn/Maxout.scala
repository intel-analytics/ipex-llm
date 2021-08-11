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
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}
import com.intel.analytics.bigdl.tensor.{DenseTensorApply, Tensor, TensorFunc6}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.serializer.{DeserializeContext, ModuleSerializable, ModuleSerializer, SerializeContext}
import com.intel.analytics.bigdl.utils.{Shape, Table}

import scala.reflect.ClassTag

/**
 * [[Maxout]] A linear maxout layer
 * Maxout layer select the element-wise maximum value of
 * maxoutNumber Linear(inputSize, outputSize) layers
 *
 * @param inputSize: the size the each input sample
 * @param outputSize: the size of the module output of each sample
 * @param maxoutNumber: number of Linear layers to use
 * @param withBias: whether use bias in Linear
 * @param wRegularizer: instance of [[Regularizer]]
 *                    (eg. L1 or L2 regularization), applied to the input weights matrices.
 * @param bRegularizer: instance of [[Regularizer]]
 *                    applied to the bias.
 * @param initWeight: initial weight
 * @param initBias: initial bias
 */
class Maxout[T: ClassTag](val inputSize: Int, val outputSize: Int, val maxoutNumber: Int,
  val withBias: Boolean = true, val wRegularizer: Regularizer[T] = null,
  val bRegularizer: Regularizer[T] = null, val initWeight: Tensor[T] = null,
                          val initBias: Tensor[T] = null)
  (implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  var layer = Sequential().add(Linear(inputSize, outputSize * maxoutNumber, withBias = withBias,
    wRegularizer = wRegularizer, bRegularizer = bRegularizer, initWeight = initWeight,
    initBias = initBias))
    .add(View(maxoutNumber, outputSize).setNumInputDims(1))
    .add(Max(1, 2))

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 2,
      s"MaxoutDense requires 2D input, but got input dim ${input.length}")
    Shape(input(0), outputSize)
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output = layer.forward(input).toTensor
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = layer.updateGradInput(input, gradOutput).toTensor
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    layer.accGradParameters(input, gradOutput)
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    layer.parameters()
  }

  override def getParametersTable(): Table = {
    layer.getParametersTable()
  }
}

object Maxout extends ModuleSerializable {
  def apply[T : ClassTag](inputSize: Int, outputSize: Int, maxoutNumber: Int,
    withBias: Boolean = true, wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null, initWeight: Tensor[T] = null, initBias: Tensor[T] = null)
    (implicit ev: TensorNumeric[T]): Maxout[T]
    = new Maxout[T](inputSize, outputSize, maxoutNumber, withBias, wRegularizer,
    bRegularizer, initWeight, initBias)

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
    val maxout = super.doLoadModule(context).asInstanceOf[Maxout[T]]
    val attrMap = context.bigdlModule.getAttrMap
    val layerAttr = attrMap.get("layer")
    maxout.layer = DataConverter.getAttributeValue(context, layerAttr).
      asInstanceOf[Sequential[T]]
    maxout
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                              maxoutBuilder : BigDLModule.Builder)
                                             (implicit ev: TensorNumeric[T]) : Unit = {
    super.doSerializeModule(context, maxoutBuilder)
    val layerBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, layerBuilder, context.moduleData.
      module.asInstanceOf[Maxout[T]].layer,
      ModuleSerializer.abstractModuleType)
    maxoutBuilder.putAttr("layer", layerBuilder.build)
  }
}
