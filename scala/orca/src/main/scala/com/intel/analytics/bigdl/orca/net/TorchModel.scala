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

import java.util
import java.util.UUID
import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.dllib.tensor.{QuantizedTensor, QuantizedType, Storage, Tensor}
import com.intel.analytics.bigdl.dllib.utils.T
import com.intel.analytics.bigdl.orca.utils.PythonInterpreter
import com.intel.analytics.bigdl.orca.net.TorchModel.TorchModel2Holder
import com.intel.analytics.bigdl.dllib.net.{NetUtils, RegistryMap, SerializationHolder}
import jep.{Jep, NDArray}
import org.apache.spark.TaskContext
import com.intel.analytics.bigdl.dllib.nn.Utils._

class TorchModel (
    private[bigdl] val modelHolder: TorchModel2Holder,
    init_weights: Array[Float])
  extends AbstractModule[Activity, Activity, Float]{
  import TorchModel._
  def postId(): String = {
    if (postfix == null) {
      postfix = Integer.toHexString(java.util.UUID.randomUUID().hashCode())
    }
    taskId() + "_" + postfix
  }

  @transient
  var postfix = Integer.toHexString(java.util.UUID.randomUUID().hashCode())
  protected var loaded = false
  @transient
  protected lazy val load = {
    PythonInterpreter.set("model_bytes", modelHolder.torchBytes)
    val loadModelCode =
      s"""
         |import torch
         |import torch.nn as nn
         |import torch.nn.functional as F
         |import torchvision
         |import io
         |from bigdl.dllib.utils.nest import ptensor_to_numpy
         |from bigdl.orca.torch.utils import trainable_param
         |from bigdl.orca.torch import zoo_pickle_module
         |
         |by = bytes(b % 256 for b in model_bytes)
         |${getName()} = torch.load(io.BytesIO(by), pickle_module=zoo_pickle_module)
         |import types
         |if isinstance(${getName()}, types.FunctionType) or \\
         |   isinstance(${getName()}, type):
         |    ${getName()} = ${getName()}()
         |""".stripMargin
    PythonInterpreter.exec(loadModelCode)
    if (extraParams.length != 0) {
      setExtraParam(extraParams)
    }
    loaded = true
    true
  }

  val weights: Tensor[Float] = Tensor[Float](Storage[Float](init_weights))
  val gradients: Tensor[Float] = Tensor[Float](weights.size())

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    (Array(weights), Array(gradients))
  }

  override def isPyTorch(): Boolean = {
    true
  }

  val setWeightCode =
    s"""
        |w = torch.Tensor(newWeight)
        |torch.nn.utils.vector_to_parameters(w, trainable_param(${getName()}))
        |
        |""".stripMargin

  def forwardCode(): String =
    s"""
       |output_${postId} = ${getName()}(input_${postId})
       |""".stripMargin

  override def updateOutput(input: Activity): Activity = {
    load
    // TODO: delete this time counting
    // _data is come from FeatureSet.
    val dataExisted = PythonInterpreter.getValue[Boolean]("'_data' in dir()")
    if (dataExisted) {
      PythonInterpreter.exec(s"input_${postId} = _data[0]")
    } else {
      // TODO: support table input
      require(input.isTensor, "only support tensor input")
      val i = input.toTensor[Float]
      if (i.nElement() == i.storage().array().length) {
        PythonInterpreter.set("nd_input_" + postId,
          new NDArray[Array[Float]](i.storage().array(), i.size(): _*))
      } else {
        // The last mini batch during evaluation is smaller.
        PythonInterpreter.set("nd_input_" + postId,
          new NDArray[Array[Float]](i.storage().array().slice(
            i.storageOffset() - 1, i.nElement()), i.size(): _*))
      }
      PythonInterpreter.exec(s"input_${postId} = torch.Tensor(nd_input_${postId})")
    }

    val forwardCode = if (train && weights.nElement() > 0) {
      PythonInterpreter.set("newWeight", new NDArray[Array[Float]](weights.storage().array()))
      PythonInterpreter.exec(setWeightCode)
      this.forwardCode
    } else {
      this.forwardCode
    }
    PythonInterpreter.exec(forwardCode)
    val outputNd = PythonFeatureSet.toArrayTensor(
      PythonInterpreter.getValue[NDArray[_]](s"ptensor_to_numpy(output_${postId})"))
    if (outputNd.length == 1) {
      if (output == null) {
        output = outputNd(0)
      } else {
        output.toTensor[Float].resizeAs(outputNd(0)).copy(outputNd(0))
      }
    } else {
      if (output != null && output.isTable &&
        output.toTable.length() == outputNd.length) {
        val t = T.array(outputNd)
        recursiveResizeAs[Float](output, t)
        recursiveCopy[Float](output, t)
      } else {
        output = T.array(outputNd)
      }
    }
    // This output_hashcode will be used in TorchLoss when compute loss.
    PythonInterpreter.exec(s"output_" +
      s"${Integer.toHexString(System.identityHashCode(output))} = output_${postId}")
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    load
    val getWeightCode =
      s"""
        |grads = []
        |none_grads = []
        |for name, param in ${getName()}.named_parameters():
        |    if param.requires_grad:
        |        if param.grad is not None:
        |            grads.append(param.grad.view(-1))
        |        else:
        |            none_grads.append(name)
        |if len(none_grads) > 0:
        |    raise Exception("Detect no gradient layer: " + " ,".join(none_grads) +
        |                    ". Please set their require_grad to False.")
        |grad = torch.nn.utils.parameters_to_vector(grads)
        |""".stripMargin
    PythonInterpreter.exec(getWeightCode)
    // TODO: Optimize this
//    val newGrads = PythonInterpreter.getValue[util.ArrayList[NDArray[Array[Float]]]](
//      "ptensor_to_numpy(grads)").asScala
//    var index = 0
//    newGrads.foreach{g =>
//      System.arraycopy(gradients.storage().array(), index,
//        g.getData(), 0, g.getDimensions()(0))
//      index += g.getDimensions()(0)
//    }
    val grad = PythonFeatureSet.ndArrayToTensor(
      PythonInterpreter.getValue("grad.data.numpy()").asInstanceOf[NDArray[_]])
    gradients.copy(grad)
    gradInput
  }

  override def zeroGradParameters(): Unit = {
    val zeroGradCode =
      s"""
        |for param in trainable_param(${this.getName()}):
        |    param.grad.fill_(0)
        |""".stripMargin
    PythonInterpreter.exec(zeroGradCode)
    super.zeroGradParameters()
  }

  override def evaluate(): this.type = {
    super.evaluate()
    load
    if (weights.nElement() != 0) {
      PythonInterpreter.set("newWeight", new NDArray[Array[Float]](weights.storage().array()))
      PythonInterpreter.exec(setWeightCode)
    }
    PythonInterpreter.exec(s"${getName()}.eval()")
    this
  }

  protected var extraParams: Array[Tensor[Float]] = Array()
  override def getExtraParameter(): Array[Tensor[Float]] = {
    if (loaded) {
      val getExtraParamCode =
        s"""
           |${getName()}_extra_parameters = []
           |for named_buffer in ${this.getName()}.named_buffers():
           |    ${getName()}_extra_parameters.append(named_buffer[1].data.numpy())
           |""".stripMargin
      PythonInterpreter.exec(getExtraParamCode)
      val extraParams = PythonInterpreter.getValue[AnyRef](s"${getName()}_extra_parameters")
      PythonFeatureSet.toArrayTensor(extraParams)
    } else {
      extraParams
    }
  }

  // TODO: change to override setExtraParameter when switch to bigdl 0.11.0
  private[bigdl] def setExtraParam(extraParams: Array[Tensor[Float]]): this.type = {
    if (loaded) {
      val params = extraParams.map(param => new NDArray[Array[Float]](param.storage().array()))
      val paramName = s"${getName()}_new_extra_param"
      val idxName = s"${getName()}_buffer_idx"
      PythonInterpreter.set(paramName, params)
      val setExtraParamCode =
        s"""
           |${idxName} = 0
           |for named_buffer in ${this.getName()}.named_buffers():
           |    named_buffer[1].copy_(
           |      torch.reshape(torch.Tensor(${paramName}[${idxName}]), named_buffer[1].size()))
           |    ${idxName} += 1
           |""".stripMargin
      PythonInterpreter.exec(setExtraParamCode)
    }
    this.extraParams = extraParams
    this
  }

  override def training(): this.type = {
    super.training()
    load
    require(weights.nElement() > 0, "We can't train an inference-only model, weights is empty.")
    PythonInterpreter.exec(s"${getName()}.train()")
    this
  }

}

object TorchModel {
  private val modelBytesRegistry = new RegistryMap[Array[Byte]]()

  @transient
  private lazy val inDriver = NetUtils.isDriver

  class TorchModel2Holder(@transient var torchBytes: Array[Byte], private var id: String)
    extends SerializationHolder {

    override def writeInternal(out: CommonOutputStream): Unit = {
      val (graphDef, _) = modelBytesRegistry.getOrCreate(id) {
        torchBytes
      }
      val len = graphDef.length
      out.writeString(id)
      if (inDriver) {
        out.writeInt(len)
        timing(s"writing ${len / 1024 / 1024}Mb torch model to stream") {
          out.write(graphDef)
        }
      } else {
        out.writeInt(0)
      }
    }

    override def readInternal(in: CommonInputStream): Unit = {
      id = in.readString()
      val (graph, _) = modelBytesRegistry.getOrCreate(id) {
        val len = in.readInt()
        assert(len >= 0, "GraphDef length should be an non-negative integer")
        val graphDef = new Array[Byte](len)
        timing("reading graph def from stream") {
          var numOfBytes = 0
          while (numOfBytes < len) {
            val read = in.read(graphDef, numOfBytes, len - numOfBytes)
            numOfBytes += read
          }
        }
        graphDef
      }

      torchBytes = graph
      id = id
    }

  }

  def apply(modelBytes: Array[Byte], weights: Array[Float]): TorchModel = {
    new TorchModel(new TorchModel2Holder(modelBytes, UUID.randomUUID().toString), weights)
  }

  /**
   * Get an inference-only model from a pytorch model.
   * This model file should be save by `torch.save` with bigdl.orca.torch.zoo_pickle_module.
   * Like: torch.save(model, "/tmp/xxx.pt", pickle_module=zoo_pickle_module)
   * @param Path model file
   * @return TorchModel
   */
  def loadModel(Path: String): TorchModel = {
    val bys = Files.readAllBytes(Paths.get(Path))
    val weights = new Array[Float](0)
    apply(bys, weights)
  }

  private[net] def taskId(): Int = TaskContext.getPartitionId()

  def isTorch(module: Module[_]): Boolean = {
    module.isInstanceOf[TorchModel]
  }
}



