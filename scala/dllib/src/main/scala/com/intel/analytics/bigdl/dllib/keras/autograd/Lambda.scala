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

package com.intel.analytics.bigdl.dllib.keras.autograd

import com.intel.analytics.bigdl.dllib.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.dllib.nn.keras.KerasLayer
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.{MultiShape, Shape, SingleShape}
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils
import com.intel.analytics.bigdl.dllib.keras.models.Model

import scala.reflect.ClassTag


private[bigdl] class LambdaTorch[T: ClassTag](val graph: Model[T])(
    implicit ev: TensorNumeric[T]) extends AbstractModule[Activity, Activity, T] {

  override def updateOutput(input: Activity): Activity = {
    output = graph.updateOutput(input)
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput = graph.updateGradInput(input, gradOutput)
    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    graph.accGradParameters(input, gradOutput)
  }
}

private[bigdl] class Lambda[T: ClassTag](val func: (List[Variable[T]]) => Variable[T],
    inputShape: Shape = null)(
    implicit ev: TensorNumeric[T]) {

  def getInputShape(): Shape = inputShape

  def inputs(nodes : ModuleNode[T]*): ModuleNode[T] = {
    val inputShape = Shape(nodes.map {node =>
    node.element.getOutputShape()
    }.toList)
    val lambda = this.create(KerasUtils.removeBatch(inputShape))
    lambda.inputs(nodes : _*)
  }

  def inputs(nodes : Array[ModuleNode[T]]): ModuleNode[T] = {
    this.inputs(nodes : _*)
  }


  // There's no batch in the inputShape
  def create(inputShape: Shape): LambdaLayer[T] = {
    val inputs = inputShape match {
      case s: SingleShape =>
        List(Variable[T](s))
      case m: MultiShape =>
        m.value.map(s => Variable[T](s))
    }
    LambdaLayer[T](inputs.toArray, outVar = func(inputs), inputShape)
  }
}

object Lambda {

  def apply[T: ClassTag](func: (List[Variable[T]]) => Variable[T], inputShape: Shape = null)(
      implicit ev: TensorNumeric[T]): Lambda[T] = {
    new Lambda(func, inputShape)
  }
}

object LambdaLayer {
  def apply[T: ClassTag](inputs: Array[Variable[T]],
  outVar: Variable[T], inputShape: Shape)(implicit ev: TensorNumeric[T]): LambdaLayer[T] = {
    new LambdaLayer[T](outVar.toGraph(inputs), inputShape)
  }
}

class LambdaLayer[T: ClassTag] private (val graph: Model[T],
    val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Activity, Activity, T](KerasUtils.addBatch(inputShape)) with Net {
  override def computeOutputShape(inputShape: Shape): Shape = {
    graph.getOutputShape()
  }

  override def doBuild(inputShape: Shape): LambdaTorch[T] = {
    new LambdaTorch[T](graph)
  }
}
