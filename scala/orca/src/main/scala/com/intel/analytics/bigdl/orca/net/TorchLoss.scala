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

import java.util.UUID

import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractCriterion, Activity}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.orca.utils.PythonInterpreter
import jep.NDArray
import org.apache.spark.TaskContext


class TorchLoss(private val criterionHolder: Array[Byte])
  extends AbstractCriterion[Activity, Activity, Float]() {
  import TorchLoss._
  def postId(): String = {
    taskId() + "_" + postfix
  }

  @transient
  lazy val postfix = Integer.toHexString(java.util.UUID.randomUUID().hashCode())

  protected lazy val loaded = {
    PythonInterpreter.set("criterion_bytes", criterionHolder)
    val loadModelCode =
      s"""
         |from pyspark.serializers import CloudPickleSerializer
         |c_by = bytes(b % 256 for b in criterion_bytes)
         |${name} = CloudPickleSerializer.loads(CloudPickleSerializer, c_by)
         |""".stripMargin
    PythonInterpreter.exec(loadModelCode)
    true
  }

  override def updateOutput(input: Activity, target: Activity): Float = {
    loaded
    // _data is come from FeatureSet.
    val dataExisted = PythonInterpreter.getValue[Boolean]("'_data' in dir()")
    if (dataExisted) {
      PythonInterpreter.exec(s"target_${postId} = _data[1]")
    } else {
      // TODO: support table target
      require(target.isTensor, "only support tensor target")
      // TODO: detect type
      val t = target.toTensor[Float]
      if (t.nElement() == t.storage().array().length) {
        PythonInterpreter.set(s"nd_target_${postId}",
          new NDArray[Array[Float]](t.storage().array(), t.size(): _*))
      } else {
        // The last mini batch during evaluation is smaller.
        PythonInterpreter.set(s"nd_target_${postId}",
          new NDArray[Array[Float]](t.storage().array().slice(
            t.storageOffset() - 1, t.nElement()), t.size(): _*))
      }
      PythonInterpreter.exec(s"target_${postId} = torch.Tensor(nd_target_${postId})")
    }
    PythonInterpreter.exec(s"loss_${postId} = ${name}(" +
      s"output_${Integer.toHexString(System.identityHashCode(input))}, target_${postId})")
    output = PythonInterpreter.getValue(s"loss_${postId}.item()").asInstanceOf[Double].toFloat
    output
  }

  override def updateGradInput(input: Activity, target: Activity): Activity = {
    // TODO: return a empty result
    val backwardCode =
      s"""
         |loss_${postId}.backward(retain_graph=True)
         |""".stripMargin
    PythonInterpreter.exec(backwardCode)
    Tensor[Float]()
  }

  protected val name =
    s"${this.getClass.getSimpleName}${Integer.toHexString(java.util.UUID.randomUUID().hashCode())}"

}

object TorchLoss{
  def apply(modelBytes: Array[Byte]): TorchLoss = {
    new TorchLoss(modelBytes)
  }

  private[net] def taskId(): Int = TaskContext.getPartitionId()
}


