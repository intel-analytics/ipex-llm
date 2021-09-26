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
package com.intel.analytics.bigdl.orca.net

import com.intel.analytics.bigdl.dllib.optim.OptimMethod
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.{EngineType, Table}
import com.intel.analytics.bigdl.orca.utils.PythonInterpreter
import jep.NDArray
import org.apache.spark.TaskContext

import scala.reflect.ClassTag
import com.intel.analytics.bigdl.dllib.keras.models.InternalOptimizerUtil
import com.intel.analytics.bigdl.orca.net.TorchOptim.DecayType

class TorchOptim[@specialized(Float, Double) T: ClassTag](
    torchOptim: Array[Byte],
    decayType: DecayType)(implicit ev: TensorNumeric[T]) extends OptimMethod[T] {
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
         |from torch.optim.optimizer import *
         |from torch.optim.lr_scheduler import _LRScheduler
         |from torch.optim.lr_scheduler import *
         |from bigdl.orca.torch import zoo_pickle_module
         |
         |optim_by = bytes(b % 256 for b in optim_bytes)
         |$name = torch.load(io.BytesIO(optim_by), pickle_module=zoo_pickle_module)
         |""".stripMargin
    PythonInterpreter.exec(loadModelCode)
    weightName = name + "_weight"
    gradientName = name + "gradient"
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
      LrScheduler
    } else if (PythonInterpreter.getValue[Boolean](s"isinstance($name, ReduceLROnPlateau)")) {
      // ReduceLROnPlateau is not subclass of LRScheduler
      require(decayType == EpochDecayByScore, "Plateau should use decayType EpochDecayByScore")
      initCode = s"""
           |$weightName = torch.tensor($weightName, requires_grad=True)
           |$weightName = torch.autograd.Variable($weightName)
           |${name}.optimizer.__init__([${weightName}], **${name}.optimizer.defaults)
           |""".stripMargin
      stepCode = s"""
           |${weightName}.grad = torch.tensor(${gradientName})
           |${name}.optimizer.step()
           |""".stripMargin
      Plateau
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
    } else {
      updateHyperParameter()
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
      case LrScheduler =>
        // TODO: multi LR support.
        PythonInterpreter.getValue[Double](s"${name}.get_last_lr()[0]")
      case Plateau =>
        if (PythonInterpreter.getValue[Boolean](s"hasattr(${name}, '_last_lr')")) {
          PythonInterpreter.getValue[Double](s"${name}._last_lr[0]")
        } else {
          PythonInterpreter.getValue[Double](s"${name}.optimizer.defaults['lr']")
        }
      case _ =>
        throw new IllegalArgumentException()
    }
  }

  override def loadFromTable(config: Table): TorchOptim.this.type = {
    this
  }

  override def updateHyperParameter(): Unit = {
    if (optimType == LrScheduler || optimType == Plateau) {
      val epoch = getEpoch(this)
      decayType match {
        case TorchOptim.EpochDecay =>
          if (lastEpoch < epoch) {
            PythonInterpreter.exec(s"${name}.step()")
            lastEpoch += 1
          }
        case TorchOptim.IterationDecay =>
          PythonInterpreter.exec(s"${name}.step()")
        case TorchOptim.EpochDecayByScore =>
          if (lastEpoch < epoch) {
            val valScore = getScore(this)
            PythonInterpreter.set("val_score", java.lang.Float.valueOf(valScore))
            PythonInterpreter.exec(s"${name}.step(val_score)")
            lastEpoch += 1
          }
      }
    }
  }

  override def getHyperParameter(): String = {
    if (optimType == LrScheduler) {
      s"Current learning rate is ${getLearningRate()}. "
    } else {
      ""
    }
  }

}

object TorchOptim{
  sealed trait OptimType
  case object LrScheduler extends OptimType
  case object Optim extends OptimType
  case object Plateau extends OptimType

  sealed trait DecayType
  case object EpochDecay extends DecayType
  case object IterationDecay extends DecayType
  case object EpochDecayByScore extends DecayType
  // TODO: Support this later.
//  case object IterationDecayByEpoch extends DecayType

  def getDecayType(decayType: String): DecayType = {
    decayType.toLowerCase() match {
      case "epochdecay" =>
        EpochDecay
      case "iterationdecay" =>
        IterationDecay
      case "epochdecaybyscore" =>
        EpochDecayByScore
//      case "iterationdecaybyepoch" =>
//        IterationDecayByEpoch
      case _ =>
        throw new IllegalArgumentException(s"unknow decay type: ${decayType}, expected:" +
          s"EpochDecay, IterationDecay, EpochDecayByScore")
    }

  }

  def apply[T: ClassTag](
        optimBytes: Array[Byte],
        decayType: String)(implicit ev: TensorNumeric[T]): TorchOptim[T] = {
    apply[T](optimBytes, getDecayType(decayType))
  }

  def apply[T: ClassTag](
        optimBytes: Array[Byte],
        decayType: DecayType)(implicit ev: TensorNumeric[T]): TorchOptim[T] = {
    new TorchOptim[T](optimBytes, decayType)
  }

  protected[net] def getEpoch[T: ClassTag](optim: TorchOptim[T]): Int = {
    // BigDL's epoch starts from 1, while torch starts from 0.
    InternalOptimizerUtil.getStateFromOptiMethod(optim)[Int]("epoch") - 1
  }

  protected[net] def getScore[T: ClassTag](optim: TorchOptim[T]): Float = {
    // BigDL's epoch starts from 1, while torch starts from 0.
    InternalOptimizerUtil.getStateFromOptiMethod(optim)[Float]("score")
  }
}
