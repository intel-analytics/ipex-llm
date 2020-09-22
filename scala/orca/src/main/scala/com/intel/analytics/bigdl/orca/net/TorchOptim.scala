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
package com.intel.analytics.zoo.pipeline.api.net

import com.intel.analytics.bigdl.optim.OptimMethod
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{EngineType, Table}
import com.intel.analytics.zoo.common.PythonInterpreter
import com.intel.analytics.zoo.feature.PythonFeatureSet
import jep.NDArray
import org.apache.spark.TaskContext

import scala.reflect.ClassTag
import com.intel.analytics.zoo.pipeline.api.keras.models.InternalOptimizerUtil

class TorchOptim[@specialized(Float, Double) T: ClassTag](
    torchOptim: Array[Byte])(implicit ev: TensorNumeric[T]) extends OptimMethod[T] {
  import TorchOptim._

  @transient
  protected val postfix = Integer.toHexString(java.util.UUID.randomUUID().hashCode())
  @transient
  protected lazy val optimType: OptimType = {
    val partId = TaskContext.getPartitionId()
    name = s"optim_${postfix}_${partId}"
    PythonInterpreter.set("optim_bytes", torchOptim)
    val currentEpoch = getEpoch(this)
    val loadModelCode =
      s"""
         |import torch
         |import io
         |from torch.optim.optimizer import Optimizer
         |from torch.optim.lr_scheduler import _LRScheduler
         |from zoo.pipeline.api.torch import zoo_pickle_module
         |
         |optim_by = bytes(b % 256 for b in optim_bytes)
         |$name = torch.load(io.BytesIO(optim_by), pickle_module=zoo_pickle_module)
         |""".stripMargin
    PythonInterpreter.exec(loadModelCode)
    weightName = name + "_weight"
    gradientName = name + "gradient"
    lrStepCode = s"""
                  |${name}.step()
                  |""".stripMargin
    if (PythonInterpreter.getValue[Boolean](s"isinstance($name, Optimizer)")) {
      initCode = s"""
           |$weightName = torch.tensor($weightName, requires_grad=True)
           |$weightName = torch.autograd.Variable($weightName)
           |${name}.__init__([${weightName}], **${name}.defaults)
           |""".stripMargin
      stepCode = s"""
           |${weightName}.grad = torch.tensor(${gradientName})
           |${name}.step()
           |""".stripMargin
      Optim
    } else if (PythonInterpreter.getValue[Boolean](s"isinstance($name, _LRScheduler)")) {
      initCode = s"""
           |$weightName = torch.tensor($weightName, requires_grad=True)
           |$weightName = torch.autograd.Variable($weightName)
           |${name}.optimizer.__init__([${weightName}], **${name}.optimizer.defaults)
           |""".stripMargin
      stepCode = s"""
           |${weightName}.grad = torch.tensor(${gradientName})
           |${name}.optimizer.step()
           |""".stripMargin
      LrSchedule
    } else {
      val unknowType = PythonInterpreter.getValue[String](s"str(type($name))")
      throw new IllegalArgumentException(s"Unknown optimizer type: " + unknowType)
    }
  }

  @transient
  protected var name = ""
  @transient
  protected var weightName = ""
  @transient
  protected var gradientName = ""
  @transient
  protected var initCode = ""
  @transient
  protected var lrStepCode = ""
  @transient
  protected var stepCode = ""
  @transient
  protected var init = false
  @transient
  protected var lastEpoch = -1

  override def optimize(
        feval: Tensor[T] => (T, Tensor[T]),
        parameter: Tensor[T]): (Tensor[T], Array[T]) = {
    optimType
    val epoch = getEpoch(this)
    val (fx, dfdx) = feval(parameter)
    if (!init) {
      lastEpoch = epoch
      PythonInterpreter.set(weightName, new NDArray[Array[Float]](
        parameter.toTensor[Float].storage().array()))
      PythonInterpreter.exec(initCode)
      init = true
    }
    if (optimType == LrSchedule && lastEpoch < epoch) {
      PythonInterpreter.exec(lrStepCode)
    }
    PythonInterpreter.set(gradientName, new NDArray[Array[Float]](
      dfdx.toTensor[Float].storage().array()))
    PythonInterpreter.exec(stepCode)
    val updatedParameter = PythonFeatureSet.ndArrayToTensor(
      PythonInterpreter.getValue(s"${weightName}.data.numpy()").asInstanceOf[NDArray[_]])
    parameter.copy(updatedParameter.toTensor[T])
    (parameter, Array(fx))

  }

  override def clearHistory(): Unit = {

  }

  override def getLearningRate(): Double = {
    optimType match {
      case Optim =>
        PythonInterpreter.getValue[Double](s"${name}.defaults['lr']")
      case lrSchedule =>
        // TODO: multi LR support.
        PythonInterpreter.getValue[Double](s"${name}.get_last_lr()[0]")
      case _ =>
        throw new IllegalArgumentException()
    }
  }

  override def loadFromTable(config: Table): TorchOptim.this.type = {
    this
  }

  override def updateHyperParameter(): Unit = {
    if (optimType == LrSchedule) {
      val epoch = getEpoch(this)
      PythonInterpreter.exec(s"${name}.step(${epoch})")
    }
  }

  override def getHyperParameter(): String = {
    if (optimType == LrSchedule) {
      s"Current learning rate is ${getLearningRate()}. "
    } else {
      ""
    }
  }

}

object TorchOptim{
  sealed trait OptimType

  case object LrSchedule extends OptimType
  case object Optim extends OptimType

  def apply[T: ClassTag](optimBytes: Array[Byte])(implicit ev: TensorNumeric[T]): TorchOptim[T] = {
    new TorchOptim[T](optimBytes)
  }

  protected[net] def getEpoch[T: ClassTag](optim: TorchOptim[T]): Int = {
    // BigDL's epoch starts from 1, while torch starts from 0.
    InternalOptimizerUtil.getStateFromOptiMethod(optim)[Int]("epoch") - 1
  }
}
