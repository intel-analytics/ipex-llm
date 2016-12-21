/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.example.loadCaffe

import java.io.{File, FileInputStream, InputStreamReader}

import caffe.Caffe
import caffe.Caffe.{LayerParameter, NetParameter, V1LayerParameter}
import com.google.protobuf.{CodedInputStream, TextFormat}
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Utils
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class CaffeLoader[@specialized(Float, Double) T: ClassTag](defName: String, modelName: String,
  matchAll: Boolean = true //if match all modules with parameters
)(implicit ev: TensorNumeric[T]) {
  private var netparam: Caffe.NetParameter = _
  private var name2LayerV1: Map[String, V1LayerParameter] = _
  private var name2LayerV2: Map[String, LayerParameter] = _

  private def loadCaffe(prototxtName: String, modelName: String): Unit = {
    if (name2LayerV2 == null) {
      netparam = loadBinary(prototxtName, modelName)
      name2LayerV2 = Map[String, LayerParameter]()
      name2LayerV1 = Map[String, V1LayerParameter]()
      import scala.collection.JavaConversions._
      // V1LayerParameter
      netparam.getLayersList.foreach(layer => {
        name2LayerV1 += (layer.getName -> layer)
      })
      // V2LayerParameter
      netparam.getLayerList.foreach(layer => {
        name2LayerV2 += (layer.getName -> layer)
      })
    }
  }

  private def loadBinary(prototxtName: String, modelName: String): Caffe.NetParameter = {
    val f = new File(prototxtName)
    assert(f.exists(), prototxtName + " does not exists")
    val reader = new InputStreamReader(new FileInputStream(f), "ASCII")
    val builder = NetParameter.newBuilder
    TextFormat.merge(reader, builder)
    println(s"start loading caffe model from $modelName")
    val cis = CodedInputStream.newInstance(new FileInputStream(modelName))
    cis.setSizeLimit(1000000000)
    builder.mergeFrom(cis)
    println("load caffe model done")
    builder.build()
  }

  private def getBlob(name: String, ind: Int): Caffe.BlobProto = {
    if (name2LayerV2(name).getBlobsCount != 0) {
      name2LayerV2(name).getBlobs(ind)
    } else if (name2LayerV1(name).getBlobsCount != 0) {
      name2LayerV1(name).getBlobs(ind)
    } else {
      null
    }
  }

  private def loadModule(name: String, destPara: Array[Tensor[T]]):
  (Tensor[T], Tensor[T]) = {
    val caffeWeight = getBlob(name, 0)
    if (caffeWeight == null) return (null, null)
    val weightList = caffeWeight.getDataList
    require(destPara != null && destPara(0).nElement() == weightList.size(),
      s"weight element must be equal in module $name")
    require(destPara(0).isContiguous())
    val weightData = destPara(0).storage().array()
    for (i <- 0 until weightList.size()) {
      weightData(i) = ev.fromType[Float](weightList.get(i))
    }

    if (destPara.length > 1) {
      val caffeBias = getBlob(name, 1)
      if (caffeBias == null) return (destPara(1), null)
      val biasList = caffeBias.getDataList
      require(destPara(1).nElement() == biasList.size(),
        s"bias element must be equal in module $name")
      require(destPara(1).isContiguous())
      val biasData = destPara(1).storage().array()
      for (i <- 0 until biasList.size()) {
        biasData(i) = ev.fromType[Float](biasList.get(i))
      }
    }
    (destPara(0), destPara(1))
  }

  /**
   * copy caffe parameters to module
   * if matchAll, throw an exception if some layers are not mapped
   *
   * @param model the model defined in big-dl
   * @return
   */
  private def copyParameters(model: Module[T]): Module[T] = {
    loadCaffe(defName, modelName)
    val namedModules = Utils.getNamedModules[T](model)

    def copyParameter(name: String, mod: Module[T]): Unit = {
      if (mod.parameters() == null) return
      if (!name2LayerV2.contains(name) && !name2LayerV1.contains(name)) {
        if (matchAll) throw new Exception(s"module $name cannot map a layer in caffe model")
        println(s"$name uses initialized parameters")
        return
      }
      val (weight, _) = loadModule(name, mod.parameters()._1)
      if (weight == null) {
        println(s"$name uses initialized parameters")
        return
      }
    }

    namedModules.foreach {
      case (name: String, mod: Module[T]) =>
        copyParameter(name, mod)
    }
    model
  }
}

object CaffeLoader {

  def load[@specialized(Float, Double) T: ClassTag](model: Module[T],
    defPath: String, modelPath: String, matchAll: Boolean = true)
    (implicit ev: TensorNumeric[T]): Module[T] = {
    val caffeLoader = new CaffeLoader[T](defPath, modelPath, matchAll)
    caffeLoader.copyParameters(model)
  }
}
