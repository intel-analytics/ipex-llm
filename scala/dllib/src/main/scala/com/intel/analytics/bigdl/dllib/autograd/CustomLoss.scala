/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api.autograd

import com.intel.analytics.bigdl.nn.abstractnn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Shape, T}
import com.intel.analytics.zoo.pipeline.api.autograd.{AutoGrad => AG}
import com.intel.analytics.zoo.pipeline.api.keras.models.Model

import scala.reflect.ClassTag

object CustomLoss {
  def apply[T: ClassTag](
      func: (Variable[T], Variable[T]) => Variable[T],
      sizeAverage: Boolean = true)(
      implicit ev: TensorNumeric[T]): TensorCriterion[T] = {
      new CustomLossWithFunc[T](func, sizeAverage)
  }
}

private[zoo] class CustomLossWithFunc[T: ClassTag](
    func: (Variable[T], Variable[T]) => Variable[T],
    sizeAverage: Boolean = true)(
    implicit ev: TensorNumeric[T]) extends CustomLoss[T](sizeAverage = sizeAverage) {
  @volatile var lossInstance: AbstractModule[Activity, Activity, T] = null
  @volatile var inputVars: Array[Variable[T]] = null

  override def doGetLoss(inputs: Array[Variable[T]]): AbstractModule[Activity, Activity, T] = {
    if (lossInstance == null) {
      synchronized {
        if (lossInstance == null) {
          val yTrue = inputs(0)
          val yPred = inputs(1)
          lossInstance = generateLossFromVars(inputs, func(yTrue, yPred))
        }
      }
    }
    lossInstance
  }

  override def getInputVars(inputShapes: Array[Shape]): Array[Variable[T]] = {
    if (inputVars == null) {
      synchronized {
        if (inputVars == null) {
          inputVars = inputShapes.map(Variable(_))
        }
      }
    }
    inputVars
  }
}

class CustomLossWithVariable[T: ClassTag](inputs: Array[Variable[T]], lossVar: Variable[T],
    sizeAverage: Boolean = true)(
    implicit ev: TensorNumeric[T]) extends CustomLoss[T](sizeAverage = sizeAverage) {
  private val lossInstance = generateLossFromVars(this.inputs, this.lossVar)

  override protected def doGetLoss(
      inputs: Array[Variable[T]]): AbstractModule[Activity, Activity, T] = lossInstance

  override protected def getInputVars(inputShapes: Array[Shape]): Array[Variable[T]] = {
    this.inputs
  }
}

abstract class CustomLoss[T: ClassTag](sizeAverage: Boolean)(
    implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {

  protected def doGetLoss(inputs: Array[Variable[T]]): AbstractModule[Activity, Activity, T]

  protected def getInputVars(inputShapes: Array[Shape]): Array[Variable[T]]

  final def getLoss(inputShapes: Array[Shape]): AbstractModule[Activity, Activity, T] = {
    val inVars = getInputVars(inputShapes)
    doGetLoss(inVars)
  }

  final def generateLossFromVars(inVars: Array[Variable[T]], outVar: Variable[T]): Model[T] = {
    if (sizeAverage) {
      AG.mean(outVar, axis = 0).toGraph(inVars)
    } else {
      outVar.toGraph(inVars)
    }
  }

  private def tensorToNonBatchShape(tensor: Tensor[T]) = {
    val sizes = tensor.size()
    Shape(sizes.slice(1, sizes.length))
  }

  /**
   * Computes the loss using input and objective function. This function
   * returns the result which is stored in the output field.
   *
   * @param yPred input of the criterion
   * @param target target or labels
   * @return the loss of the criterion
   */
  override def updateOutput(yPred: Tensor[T], target: Tensor[T]): T = {
    val yPredT = yPred.toTensor
    val yTrueT = target.toTensor
    val nonBatchShape = tensorToNonBatchShape(yPred)
    val loss = getLoss(Array(nonBatchShape, nonBatchShape))
    val result = loss.forward(T(yTrueT, yPredT)).toTensor[T]
    require(result.isScalar,
      s"The loss should be scalar, but got result with shape: [${result.size().mkString(", ")}]")
    result.value()
  }

  /**
   * Computing the gradient of the criterion with respect to its own input. This is returned in
   * gradInput. Also, the gradInput state variable is updated accordingly.
   *
   * @param yPred input data
   * @param yTrue target data / labels
   * @return gradient of input
   */
  override def updateGradInput(yPred: Tensor[T], yTrue: Tensor[T]): Tensor[T] = {
    val nonBatchShape = tensorToNonBatchShape(yPred)
    val loss = getLoss(Array(nonBatchShape, nonBatchShape))
    val result = loss.updateGradInput(
      T(yTrue, yPred), Tensor[T](1).fill(ev.one))
    // we only respect the input grad
    result.toTable.get[Tensor[T]](2).get
  }
}
