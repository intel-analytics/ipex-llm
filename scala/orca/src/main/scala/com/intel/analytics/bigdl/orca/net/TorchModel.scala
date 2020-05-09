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

import java.util
import java.util.UUID

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.{QuantizedTensor, QuantizedType, Storage, Tensor}
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.common.PythonInterpreter
import com.intel.analytics.zoo.feature.PythonLoaderFeatureSet
import com.intel.analytics.zoo.pipeline.api.net.TorchModel.TorchModel2Holder
import jep.{Jep, NDArray}

import scala.collection.JavaConverters._

import scala.reflect.ClassTag
// TODO: support Train function
class TorchModel private(private val modelHolder: TorchModel2Holder, init_weights: Array[Float])
  extends AbstractModule[Activity, Activity, Float]{
  import TorchModel._

  protected lazy val loaded = {
    PythonInterpreter.set("model_bytes", modelHolder.torchBytes)
    val loadModelCode =
      s"""
         |import torch
         |import torch.nn as nn
         |import torch.nn.functional as F
         |import torchvision
         |from zoo.util.nest import ptensor_to_numpy
         |
         |from pyspark.serializers import CloudPickleSerializer
         |by = bytes(b % 256 for b in model_bytes)
         |${getName()} = CloudPickleSerializer.loads(CloudPickleSerializer, by)
         |""".stripMargin
    PythonInterpreter.exec(loadModelCode)
    true
  }

  val weights: Tensor[Float] = Tensor[Float](Storage[Float](init_weights))
  val gradients: Tensor[Float] = Tensor[Float](weights.size())

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    (Array(weights), Array(gradients))
  }

  val setWeightCode =
    s"""
        |w = torch.Tensor(newWeight)
        |torch.nn.utils.vector_to_parameters(w, ${getName()}.parameters())
        |
        |""".stripMargin

  val forwardCode =
    s"""
       |output = ${getName()}(input)
       |""".stripMargin

  override def updateOutput(input: Activity): Activity = {
    loaded
    // TODO: delete this time counting
    val startTime = System.nanoTime()
    // data is come from FeatureSet.
    val dataExisted = PythonInterpreter.getValue[Boolean]("'data' in dir()")
    if (dataExisted) {
      PythonInterpreter.exec("input = data[0]")
    } else {
      // TODO: support table input
      require(input.isTensor, "only support tensor input")
      val i = input.toTensor[Float]
      PythonInterpreter.set("nd_input",
        new NDArray[Array[Float]](i.storage().array(), i.size(): _*))
      PythonInterpreter.exec("input = torch.Tensor(nd_input)")
    }

    val forwardCode = if (train) {
      PythonInterpreter.set("newWeight", new NDArray[Array[Float]](weights.storage().array()))
      PythonInterpreter.exec(setWeightCode)
      println(s"setWeight time is ${(System.nanoTime() - startTime) / 1e9}")
      this.forwardCode
    } else {
      this.forwardCode
    }
    PythonInterpreter.exec(forwardCode)
    println(s"run forward cost: ${(System.nanoTime() - startTime) / 1e9}")
    val outputNd = PythonLoaderFeatureSet.toArrayTensor(
      PythonInterpreter.getValue[NDArray[_]]("ptensor_to_numpy(output.data)"))
    if (outputNd.length == 1) {
      output = outputNd(0)
    } else {
      output = T(outputNd)
    }
    println(s"forward total cost: ${(System.nanoTime() - startTime) / 1e9}")
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    loaded
    val startTime = System.nanoTime()
    val backwardCode =
      s"""
        |loss.backward(retain_graph=True)
        |""".stripMargin
    PythonInterpreter.exec(backwardCode)
    println(s"run backward cost: ${(System.nanoTime() - startTime) / 1e9}")
    val getWeightCode =
      s"""
        |grads=[]
        |for param in ${getName()}.parameters():
        |    grads.append(param.grad.view(-1))
        |""".stripMargin
    PythonInterpreter.exec(getWeightCode)
    val newGrads = PythonInterpreter.getValue[util.ArrayList[NDArray[Array[Float]]]](
      "ptensor_to_numpy(grads)").asScala
    var index = 0
    newGrads.foreach{g =>
      System.arraycopy(gradients.storage().array(), index,
        g.getData(), 0, g.getDimensions()(0))
      index += g.getDimensions()(0)
    }
    println(s"backward total cost: ${(System.nanoTime() - startTime) / 1e9}")
    gradInput
  }

  override def zeroGradParameters(): Unit = {
    val zeroGradCode =
      s"""
        |for param in ${this.getName()}.parameters():
        |    param.grad.fill_(0)
        |""".stripMargin
    PythonInterpreter.exec(zeroGradCode)
    super.zeroGradParameters()
  }

  override def evaluate(): this.type = {
    super.evaluate()
    PythonInterpreter.set("newWeight", new NDArray[Array[Float]](weights.storage().array()))
    PythonInterpreter.exec(setWeightCode)
    PythonInterpreter.exec(s"${getName()}.eval()")
    this
  }

  override def training(): this.type = {
    super.training()
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
}

