/*
 * Copyright 2021 The BigDL Authors.
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

package com.intel.analytics.bigdl.ppml.utils

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.keras.models.InternalOptimizerUtil.getParametersFromModel
import com.intel.analytics.bigdl.dllib.utils.T
import com.intel.analytics.bigdl.ppml.FLClient
import com.intel.analytics.bigdl.ppml.common.{FLPhase, Storage}
import com.intel.analytics.bigdl.ppml.generated.FlBaseProto
import com.intel.analytics.bigdl.ppml.generated.FlBaseProto._
import com.intel.analytics.bigdl.dllib.utils.{Table => DllibTable}
import com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.{BoostEval, PredictResponse, TreePredict}
import org.apache.logging.log4j.LogManager

import scala.reflect.ClassTag
import scala.util.Random
import scala.collection.JavaConversions._
import scala.collection.JavaConverters._


object ProtoUtils {
  private val logger = LogManager.getLogger(getClass)
  def outputTargetToTableProto(output: Activity,
                               target: Activity,
                               meta: MetaData = null): TensorMap = {
    val tensorProto = toFloatTensor(output.toTensor[Float])

    val builder = TensorMap.newBuilder
      .putTensors("output", tensorProto)
    if (meta != null) {
      builder.setMetaData(meta)
    }

    if (target != null) {
      val targetTensor = toFloatTensor(target.toTensor[Float])
      builder.putTensors("target", targetTensor)
    }
    builder.build()
  }
  def tableProtoToOutputTarget(storage: Storage[TensorMap]): (DllibTable, Tensor[Float]) = {
    val aggData = protoTableMapToTensorIterableMap(storage.clientData)
    val target = Tensor[Float]()
    if (aggData.contains("target")) {
      val t = aggData("target").head
      target.resizeAs(t).copy(t)
    }
    // TODO: multiple input
    val outputs = aggData.filter(_._1 != "target")
    require(outputs.size == 1)

    (T.seq(outputs.values.head.toSeq), target)
  }
  def protoTableMapToTensorIterableMap(inputMap: java.util.Map[String, FlBaseProto.TensorMap]):
    Map[String, Iterable[Tensor[Float]]] = {
    inputMap.asScala.mapValues(_.getTensorsMap).values
      .flatMap(_.asScala).groupBy(_._1)
      .map{data =>
        (data._1, data._2.map {v =>
          val data = v._2.getTensorList.asScala.toArray.map(_.toFloat)
          val shape = v._2.getShapeList.asScala.toArray.map(_.toInt)
          Tensor[Float](data, shape)
        })
      }
  }

  def toFloatTensor(data: Array[Float], shape: Array[Int]): FloatTensor = {
    FloatTensor
      .newBuilder()
      .addAllTensor(data.map(float2Float).toIterable.asJava)
      .addAllShape(shape
        .map(int2Integer).toIterable.asJava)
      .build()
  }
  def toFloatTensor(t: Tensor[Float]): FloatTensor = {
    FloatTensor
      .newBuilder()
      .addAllTensor(t.toTensor[Float].contiguous().storage()
        .array().slice(t.storageOffset() - 1, t.storageOffset() - 1 + t.nElement())
        .map(float2Float).toIterable.asJava)
      .addAllShape(t.toTensor[Float].size()
        .map(int2Integer).toIterable.asJava)
      .build()
  }
  def toFloatTensor(data: Array[Float]): FloatTensor = {
    toFloatTensor(data, Array(data.length))
  }

  def getModelWeightTable(model: Module[Float], version: Int, name: String = "test") = {
    val weights = getParametersFromModel(model)._1
    val metadata = MetaData.newBuilder
      .setName(name).setVersion(version).build
    val tensor =
      FloatTensor.newBuilder()
        .addAllTensor(weights.storage.toList.map(v => float2Float(v)))
        .addAllShape(weights.size.toList.map(v => int2Integer(v)))
        .build()
    val metamodel = TensorMap.newBuilder
      .putTensors("weights", tensor)
      .setMetaData(metadata)
      .build
    metamodel
  }


  def updateModel(model: Module[Float],
                  modelData: TensorMap): Unit = {
    val weigthBias = modelData.getTensorsMap.get("weights")
    val data = weigthBias.getTensorList.asScala.map(v => Float2float(v)).toArray
    val shape = weigthBias.getShapeList.asScala.map(v => Integer2int(v)).toArray
    val tensor = Tensor(data, shape)
    getParametersFromModel(model)._1.copy(tensor)
  }

  def getTensor(name: String, modelData: TensorMap): Tensor[Float] = {
    val dataMap = modelData.getTensorsMap.get(name)
    val data = dataMap.getTensorList.asScala.map(Float2float).toArray
    val shape = dataMap.getShapeList.asScala.map(Integer2int).toArray
    Tensor[Float](data, shape)
  }



  def randomSplit[T: ClassTag](weight: Array[Float],
                               data: Array[T],
                               seed: Int = 1): Array[Array[T]] = {
    val random = new Random(seed = seed)
    val lens = weight.map(v => (v * data.length).toInt)
    lens(lens.length - 1) = data.length - lens.slice(0, lens.length - 1).sum
    val splits = lens.map(len => new Array[T](len))
    val counts = lens.map(_ => 0)
    data.foreach{d =>
      var indx = random.nextInt(weight.length)
      while(counts(indx) == lens(indx)){
        indx = (indx + 1) % weight.length
      }
      splits(indx)(counts(indx)) = d
      counts(indx) += 1
    }
    splits
  }

  def toBoostEvals(localPredicts: Array[Map[String, Array[Boolean]]]): List[BoostEval] = {
    // Sorted by treeID
    localPredicts.map{predict =>
      BoostEval.newBuilder()
        .addAllEvaluates(predict.toSeq.sortBy(_._1).map(p => {
          TreePredict.newBuilder().setTreeID(p._1)
            .addAllPredicts(p._2.map(boolean2Boolean).toList.asJava)
            .build()
        }).toList.asJava)
        .build()
    }.toList
  }
  def toArrayFloat(response: PredictResponse): Array[Float] = {
    response.getData.getTensorsMap.get("predictResult")
      .getTensorList.asScala.toArray.map(_.toFloat)
  }

  def almostEqual(v1: Float, v2: Float): Boolean = {
    if (math.abs(v1 - v2) <= 1e-1f)
      true
    else
      false
  }
}
