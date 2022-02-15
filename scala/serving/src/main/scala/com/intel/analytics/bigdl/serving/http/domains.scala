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

package com.intel.analytics.bigdl.serving.http

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}
import java.util
import java.util.concurrent.LinkedBlockingQueue

import scala.concurrent.Await
import java.util.{HashMap, UUID}
import java.nio.file.{Files, Paths}

import akka.actor.{ActorRef, Props}
import akka.pattern.ask
import com.codahale.metrics.Timer
import com.intel.analytics.bigdl.serving.utils.Conventions
import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.complex._
import org.apache.arrow.vector.dictionary.DictionaryProvider
import org.apache.arrow.vector.ipc.{ArrowStreamReader, ArrowStreamWriter}
import org.apache.arrow.vector.types.Types.MinorType
import org.apache.arrow.vector.types.pojo.{ArrowType, Field, FieldType, Schema}
import org.apache.arrow.vector._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.concurrent.duration.DurationInt
import com.fasterxml.jackson.annotation.{JsonSubTypes, JsonTypeInfo}
import com.fasterxml.jackson.annotation.JsonSubTypes.Type
import com.fasterxml.jackson.core.JsonParser
import com.fasterxml.jackson.databind.module.SimpleModule
import com.fasterxml.jackson.databind.node.{ArrayNode, ObjectNode, TextNode}
import com.fasterxml.jackson.databind.{DeserializationContext, JsonDeserializer, JsonNode, ObjectMapper}
import com.google.common.collect.ImmutableList
import com.google.common.util.concurrent.RateLimiter
import com.intel.analytics.bigdl.dllib.feature.image.OpenCVMethod
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.T
import com.intel.analytics.bigdl.serving.http.FrontEndApp._
import com.intel.analytics.bigdl.orca.inference.InferenceModel
import org.opencv.imgcodecs.Imgcodecs
import redis.clients.jedis.JedisPoolConfig
// import com.intel.analytics.bigdl.serving.ClusterServing
import com.intel.analytics.bigdl.serving.serialization.StreamSerializer
import org.slf4j.LoggerFactory
import redis.clients.jedis.JedisPool

sealed trait ServingMessage

case class PredictionInputMessage(inputs: Seq[PredictionInput]) extends ServingMessage

case class PredictionInputFlushMessage() extends ServingMessage

case class PredictionQueryMessage(ids: Seq[String]) extends ServingMessage

case class SecuredModelSecretSaltMessage(secret: String, salt: String) extends ServingMessage

case class PredictionQueryWithTargetMessage(query: PredictionQueryMessage, target: ActorRef)
  extends ServingMessage

object PredictionInputMessage {
  def apply(input: PredictionInput): PredictionInputMessage =
    PredictionInputMessage(Seq(input))
}

sealed trait PredictionInput {
  def getId(): String

  def toHash(): HashMap[String, String]

  def toHashByStream(): HashMap[String, String]
}

case class BytesPredictionInput(uuid: String, bytesStr: String) extends PredictionInput {
  override def getId(): String = this.uuid

  def toMap(): Map[String, String] = Map("uuid" -> uuid, "bytesStr" -> bytesStr)

  override def toHash(): HashMap[String, String] = {
    val hash = new HashMap[String, String]()
    hash.put("uri", uuid)
    hash.put("data", bytesStr)
    hash
  }

  override def toHashByStream(): util.HashMap[String, String] = {
    val hash = new HashMap[String, String]()
    hash
  }
}

object BytesPredictionInput {
  def apply(str: String): BytesPredictionInput =
    BytesPredictionInput(UUID.randomUUID().toString, str)
}

case class InstancesPredictionInput(uuid: String, instances: Instances)
  extends PredictionInput with Supportive {
  override def getId(): String = this.uuid

  override def toHash(): HashMap[String, String] = {
    val hash = new HashMap[String, String]()
    val bytes = instances.toArrow()
    val b64 = java.util.Base64.getEncoder.encodeToString(bytes)
    hash.put("uri", uuid)
    hash.put("data", b64)
    hash
  }

  override def toHashByStream(): HashMap[String, String] = {
    val hash = new HashMap[String, String]()
    val bytes = StreamSerializer.objToBytes(instances)
    val b64 = java.util.Base64.getEncoder.encodeToString(bytes)
    hash.put("uri", uuid)
    hash.put("data", b64)
    hash.put("serde", "stream")
    hash
  }
}

object InstancesPredictionInput {
  def apply(instances: Instances): InstancesPredictionInput =
    InstancesPredictionInput(UUID.randomUUID().toString, instances)
}

case class PredictionOutput[Type](uuid: String, result: Type)

class ImageFeature(val b64: String)

case class SparseTensor[T](shape: List[Int], data: List[T], indices: List[List[Int]])

case class Instances(instances: List[mutable.LinkedHashMap[String, Any]]) {
  if (instances == null) {
    throw new ServingRuntimeException("no instance can be found, " +
      "please check your input or json(keyword should be 'instances')")
  }

  def makeActivities(features: Array[String]): Seq[Activity] = {
    if (instances.isEmpty) {
      Seq[Activity]()
    } else {
      instances.flatMap(insMap => {
        val oneInsMap = insMap.map {
          kv => {
            if (!features.contains(kv._1)) {
              throw ServingRuntimeException("Cannot Find the feature " + kv._1 + ", Please Check " +
                "Your Input Data")
            }
            kv._2 match {
              case (value: List[_]) =>
                transferListToTensor(value)
              case (value: Map[_, _]) =>
                val map = value.asInstanceOf[Map[String, Any]]
                var result: Tensor[Float] = null
                for ((key, value) <- map) {
                  if (key == "b64") {
                    val byteBuffer = timing("decode")() {
                      java.util.Base64.getDecoder.decode(value.toString)
                    }
                    val mat = timing("load byte buffer")() {
                      OpenCVMethod.fromImageBytes(byteBuffer, Imgcodecs.CV_LOAD_IMAGE_UNCHANGED)
                    }
                    val (height, width, channel) = (mat.height(), mat.width(), mat.channels())
                    val arrayBuffer = new Array[Float](height * width * channel)
                    timing("to float pixels")() {
                      OpenCVMat.toFloatPixels(mat, arrayBuffer)
                    }
                    val imageTensor = timing("retrive image tensors")() {
                      Tensor[Float](arrayBuffer, Array(1, 1, height, width, channel))
                    }
                    result = imageTensor
                  }
                }
                result
            }
          }

        }.toList
        timing("convert to Seq of T array")() {
          Seq(T.array(oneInsMap.toArray))
        }
      })
    }
  }


  private def transferListToTensor(eleList: List[_]): Tensor[Float] = {
    if (eleList.isEmpty) {
      return Tensor[Float]()
    }
    eleList.head match {
      case _: Int =>
        val tensor = Tensor[Float](eleList.length)
        (1 to eleList.length).foreach(i => {
          tensor.setValue(i, eleList(i - 1).asInstanceOf[Int].toFloat)
        })
        tensor
      case _: Float =>
        val tensor = Tensor[Float](eleList.length)
        (1 to eleList.length).foreach(i => {
          tensor.setValue(i, eleList(i - 1).asInstanceOf[Float])
        })
        tensor
      case _: Double =>
        val tensor = Tensor[Float](eleList.length)
        (1 to eleList.length).foreach(i => {
          tensor.setValue(i, eleList(i - 1).asInstanceOf[Double].toFloat)
        })
        tensor
      case _: List[_] =>
        val dimList = new ArrayBuffer[Int](2)
        val length = eleList.size
        val width = eleList.head.asInstanceOf[List[_]].size
        var tmpList = eleList.head.asInstanceOf[List[_]]
        dimList.append(length)
        dimList.append(width)
        var size = length * width
        while (tmpList.head.isInstanceOf[List[_]]) {
          tmpList = tmpList.head.asInstanceOf[List[_]]
          dimList.append(tmpList.size)
          size *= tmpList.size
        }
        val tensor = Tensor[Float](dimList: _*)
        (1 to size).foreach(i => {
          dimList.length match {
            case 2 =>
              val dimX = (i - 1) / dimList(1)
              val dimY = (i - 1) % dimList(1)
              val value = eleList(dimX).asInstanceOf[List[_]](dimY) match {
                case value : Int => value.toFloat
                case value : Double => value.toFloat
                case value : Float => value
              }
              tensor.setValue(dimX + 1, dimY + 1, value)
            case 3 =>
              val dimX = (i - 1) / (dimList(1) * dimList(2))
              val dimY = ((i - 1) % (dimList(1) * dimList(2))) / dimList(2)
              val dimZ = (i - 1) %  dimList(2)
              val value = eleList(dimX).asInstanceOf[List[_]](dimY).asInstanceOf[List[_]](dimZ)
              match {
                case value : Int => value.toFloat
                case value : Double => value.toFloat
                case value : Float => value
              }
              tensor.setValue(dimX + 1, dimY + 1, dimZ + 1, value)
            case 4 =>
              val dimX = (i - 1) / (dimList(1) * dimList(2)* dimList(3))
              val dimY = ((i - 1) % (dimList(1) * dimList(2)* dimList(3))) /
                (dimList(2) * dimList(3))
              val dimZ = ((i - 1) % (dimList(2) * dimList(3))) / dimList(3)
              val dimZA = (i - 1) % dimList(3)
              val value = eleList(dimX).asInstanceOf[List[_]](dimY).asInstanceOf[List[_]](dimZ)
                .asInstanceOf[List[_]](dimZA) match {
                case value : Int => value.toFloat
                case value : Double => value.toFloat
                case value : Float => value
              }
              tensor.setValue(dimX + 1, dimY + 1, dimZ + 1, dimZA + 1, value)
            case 5 =>
              val dimX = (i - 1) / (dimList(1) * dimList(2)* dimList(3) * dimList(4))
              val dimY = ((i - 1) % (dimList(1) * dimList(2)* dimList(3) * dimList(4))) /
                (dimList(2)* dimList(3) * dimList(4))
              val dimZ = ((i - 1) % (dimList(2)* dimList(3) * dimList(4))) / (dimList(3) *
                dimList(4))
              val dimZA = ((i - 1) % (dimList(3) * dimList(4))) / dimList(4)
              val dimZB = (i - 1) % dimList(4)
              val value = eleList(dimX).asInstanceOf[List[_]](dimY).asInstanceOf[List[_]](dimZ)
                .asInstanceOf[List[_]](dimZA).asInstanceOf[List[_]](dimZB) match {
                case value : Int => value.toFloat
                case value : Double => value.toFloat
                case value : Float => value
              }
              tensor.setValue(dimX + 1, dimY + 1, dimZ + 1, dimZA + 1, dimZB + 1, value)
          }
        })
        tensor
    }

  }

  def constructTensors(): Seq[mutable.LinkedHashMap[String, (
    (mutable.ArrayBuffer[Int], Any), (mutable.ArrayBuffer[Int], mutable.ArrayBuffer[Any])
    )]] = {
    instances.map(instance => {
      instance.map(i => {
        val key = i._1
        val value = i._2
        if (value.isInstanceOf[SparseTensor[_]]) {
          val sparseTensor = value.asInstanceOf[SparseTensor[_]]
          val shape = mutable.ArrayBuffer[Int]()
          shape.appendAll(sparseTensor.shape)
          val data = mutable.ArrayBuffer[Any]()
          data.appendAll(sparseTensor.data)
          val indicesTensor = Instances.transferListToTensor(sparseTensor.indices)
          key -> ((shape, data), indicesTensor)
        } else {
          val tensor =
            if (value.isInstanceOf[List[_]]) {
              Instances.transferListToTensor(value)
            } else {
              (new mutable.ArrayBuffer[Int](0), value)
            }
          key -> (tensor, Instances.transferListToTensor(List()))
        }
      })
    })
  }

  def makeSchema(
                  tensors: Seq[mutable.LinkedHashMap[String, (
                    (mutable.ArrayBuffer[Int], Any),
                      (mutable.ArrayBuffer[Int], mutable.ArrayBuffer[Any])
                    )]]): Schema = {
    assert(instances.size > 0, "must have instances, and each should have the same schema")
    val sample = tensors(0)
    val key_fields = sample.map(s => (s._1, s._2))
    val childrenBuilder = ImmutableList.builder[Field]()
    key_fields.map(key_field => {
      val key = key_field._1
      val values = key_field._2._1
      val indices = key_field._2._2
      val shape = values._1
      val data = values._2
      val (isList, fieldSampe) = data.isInstanceOf[ArrayBuffer[_]] match {
        case true => (true, data.asInstanceOf[ArrayBuffer[_]](0))
        case false => (false, data)
      }
      if (fieldSampe.isInstanceOf[Int]) {
        val field = if (isList) {
          val shapeSize = shape.asInstanceOf[ArrayBuffer[_]].size
          val shapeList = new util.ArrayList[Field]()
          shapeList.add(new Field("", FieldType.nullable(Conventions.ARROW_INT), null))
          val shapeField = new Field("shape",
            FieldType.nullable(new ArrowType.List()), shapeList)
          val dataSize = data.asInstanceOf[ArrayBuffer[_]].size
          val dataList = new util.ArrayList[Field]()
          dataList.add(new Field("", FieldType.nullable(Conventions.ARROW_INT), null))
          val dataField = new Field("data",
            FieldType.nullable(new ArrowType.List()), dataList)
          val indicesShapeList = new util.ArrayList[Field]()
          indicesShapeList.add(new Field("", FieldType.nullable(Conventions.ARROW_INT), null))
          val indicesShapeField = new Field("indiceShape",
            FieldType.nullable(new ArrowType.List()), indicesShapeList)
          val indicesDataList = new util.ArrayList[Field]()
          indicesDataList.add(new Field("", FieldType.nullable(Conventions.ARROW_INT), null))
          val indicesDataField = new Field("indiceData",
            FieldType.nullable(new ArrowType.List()), indicesDataList)

          val tensorFieldList = new util.ArrayList[Field]()
          tensorFieldList.add(shapeField)
          tensorFieldList.add(dataField)
          tensorFieldList.add(indicesShapeField)
          tensorFieldList.add(indicesDataField)
          new Field(key, FieldType.nullable(new ArrowType.Struct()), tensorFieldList)
        } else {
          new Field(key, FieldType.nullable(Conventions.ARROW_INT), null)
        }
        childrenBuilder.add(field)
      } else if (fieldSampe.isInstanceOf[Float] || fieldSampe.isInstanceOf[Double]) {
        val field = if (isList) {
          val shapeSize = shape.asInstanceOf[ArrayBuffer[_]].size
          val shapeList = new util.ArrayList[Field]()
          shapeList.add(new Field("", FieldType.nullable(Conventions.ARROW_INT), null))
          val shapeField = new Field("shape",
            FieldType.nullable(new ArrowType.List()), shapeList)
          val dataSize = data.asInstanceOf[ArrayBuffer[_]].size
          val dataList = new util.ArrayList[Field]()
          dataList.add(new Field("", FieldType.nullable(Conventions.ARROW_FLOAT), null))
          val dataField = new Field("data",
            FieldType.nullable(new ArrowType.List()), dataList)
          val indicesShapeList = new util.ArrayList[Field]()
          indicesShapeList.add(new Field("", FieldType.nullable(Conventions.ARROW_INT), null))
          val indicesShapeField = new Field("indiceShape",
            FieldType.nullable(new ArrowType.List()), indicesShapeList)
          val indicesDataList = new util.ArrayList[Field]()
          indicesDataList.add(new Field("", FieldType.nullable(Conventions.ARROW_INT), null))
          val indicesDataField = new Field("indiceData",
            FieldType.nullable(new ArrowType.List()), indicesDataList)

          val tensorFieldList = new util.ArrayList[Field]()
          tensorFieldList.add(shapeField)
          tensorFieldList.add(dataField)
          tensorFieldList.add(indicesShapeField)
          tensorFieldList.add(indicesDataField)
          new Field(key, FieldType.nullable(new ArrowType.Struct()), tensorFieldList)
        } else {
          new Field(key, FieldType.nullable(Conventions.ARROW_FLOAT), null)
        }
        childrenBuilder.add(field)
      } else if (fieldSampe.isInstanceOf[String]) {
        val field = if (isList) {
          val shapeSize = shape.asInstanceOf[ArrayBuffer[_]].size
          val shapeList = new util.ArrayList[Field]()
          shapeList.add(new Field("", FieldType.nullable(Conventions.ARROW_INT), null))
          val shapeField = new Field("shape",
            FieldType.nullable(new ArrowType.List()), shapeList)
          val dataSize = data.asInstanceOf[ArrayBuffer[_]].size
          val dataList = new util.ArrayList[Field]()
          dataList.add(new Field("", FieldType.nullable(Conventions.ARROW_UTF8), null))
          val dataField = new Field("data",
            FieldType.nullable(new ArrowType.List()), dataList)
          val indicesShapeList = new util.ArrayList[Field]()
          indicesShapeList.add(new Field("", FieldType.nullable(Conventions.ARROW_INT), null))
          val indicesShapeField = new Field("indiceShape",
            FieldType.nullable(new ArrowType.List()), indicesShapeList)
          val indicesDataList = new util.ArrayList[Field]()
          indicesDataList.add(new Field("", FieldType.nullable(Conventions.ARROW_INT), null))
          val indicesDataField = new Field("indiceData",
            FieldType.nullable(new ArrowType.List()), indicesDataList)

          val tensorFieldList = new util.ArrayList[Field]()
          tensorFieldList.add(shapeField)
          tensorFieldList.add(dataField)
          tensorFieldList.add(indicesShapeField)
          tensorFieldList.add(indicesDataField)
          new Field(key, FieldType.nullable(new ArrowType.Struct()), tensorFieldList)
        } else {
          new Field(key, FieldType.nullable(Conventions.ARROW_UTF8), null)
        }
        childrenBuilder.add(field)
      }
    })
    new Schema(childrenBuilder.build(), null)
  }

  def toArrow(): Array[Byte] = {
    val tensors = constructTensors()
    val schema = makeSchema(tensors)
    val vectorSchemaRoot = VectorSchemaRoot.create(schema, new RootAllocator(Integer.MAX_VALUE))
    val provider = new DictionaryProvider.MapDictionaryProvider
    val byteArrayOutputStream = new ByteArrayOutputStream()
    val arrowStreamWriter = new ArrowStreamWriter(vectorSchemaRoot, provider, byteArrayOutputStream)
    arrowStreamWriter.start()
    for (i <- 0 until tensors.size) {
      val map = tensors(i)
      vectorSchemaRoot.setRowCount(1)
      map.map(sample => {
        val key = sample._1
        val tensor = sample._2._1
        val indices = sample._2._2
        val fieldVector = vectorSchemaRoot.getVector(key)
        fieldVector.setInitialCapacity(1)
        fieldVector.allocateNew()

        val minorType = fieldVector.getMinorType()
        minorType match {
          case MinorType.INT =>
            fieldVector.asInstanceOf[IntVector].setSafe(0, tensor._2.asInstanceOf[Int])
            fieldVector.setValueCount(1)
          case MinorType.FLOAT4 =>
            tensor._2.isInstanceOf[Float] match {
              case true => fieldVector.asInstanceOf[Float4Vector]
                .setSafe(0, tensor._2.asInstanceOf[Float])
              case false => fieldVector.asInstanceOf[Float4Vector]
                .setSafe(0, tensor._2.asInstanceOf[Double].toFloat)
            }
            fieldVector.setValueCount(1)
          case MinorType.VARCHAR =>
            val varCharVector = fieldVector.asInstanceOf[VarCharVector]
            val bytes = tensor._2.asInstanceOf[String].getBytes
            varCharVector.setSafe(0, bytes)
            fieldVector.setValueCount(1)
          case MinorType.VARBINARY =>
            val varBinaryVector = fieldVector.asInstanceOf[VarBinaryVector]
            val bytes = tensor._2.asInstanceOf[String].getBytes
            varBinaryVector.setIndexDefined(0)
            varBinaryVector.setValueLengthSafe(0, bytes.length)
            varBinaryVector.setSafe(0, bytes)
            fieldVector.setValueCount(1)
          case MinorType.STRUCT =>
            val shape = tensor._1
            val data = tensor._2
            val indicesShape = indices._1
            val indicesData = indices._2
            val structVector = fieldVector.asInstanceOf[StructVector]
            val shapeVector = structVector.getChild("shape").asInstanceOf[ListVector]
            val dataVector = structVector.getChild("data").asInstanceOf[ListVector]
            val indicesShapeVector = structVector.getChild("indiceShape").asInstanceOf[ListVector]
            val indicesDataVector = structVector.getChild("indiceData").asInstanceOf[ListVector]
            val shapeDataVector = shapeVector.getDataVector
            val dataDataVector = dataVector.getDataVector
            val indicesShapeDataVector = indicesShapeVector.getDataVector
            val indicesDataDataVector = indicesDataVector.getDataVector

            shapeVector.allocateNew()
            val shapeSize = shape.size
            val shapeIntVector = shapeDataVector.asInstanceOf[IntVector]
            for (j <- 0 until shapeSize) {
              shapeVector.startNewValue(j)
              shapeIntVector.setSafe(j, shape(j))
              shapeVector.endValue(j, 1)
            }
            shapeVector.setValueCount(shapeSize)
            shapeIntVector.setValueCount(shapeSize)

            dataVector.allocateNew()
            dataDataVector.getMinorType match {
              case MinorType.INT =>
                val dataIntVector = dataDataVector.asInstanceOf[IntVector]
                val datas = data.asInstanceOf[ArrayBuffer[Int]]
                val dataSize = datas.size
                for (j <- 0 until dataSize) {
                  dataVector.startNewValue(j)
                  dataIntVector.setSafe(j, datas(j))
                  dataVector.endValue(j, 1)
                }
                dataIntVector.setValueCount(dataSize)
                dataVector.setValueCount(dataSize)
              case MinorType.FLOAT4 =>
                val dataFloatVector = dataDataVector.asInstanceOf[Float4Vector]
                val dataBuffer = data.asInstanceOf[ArrayBuffer[_]]
                dataBuffer.size > 0 match {
                  case true =>
                    val dataSample = dataBuffer(0)
                    val dataSize = dataBuffer.size
                    for (j <- 0 until dataSize) {
                      dataBuffer(j).isInstanceOf[Float] match {
                        case true =>
                          dataVector.startNewValue(j)
                          dataFloatVector.setSafe(j, dataBuffer(j).asInstanceOf[Float])
                          dataVector.endValue(j, 1)
                        case false =>
                          dataVector.startNewValue(j)
                          dataFloatVector.setSafe(j, dataBuffer(j).asInstanceOf[Double].toFloat)
                          dataVector.endValue(j, 1)
                      }
                    }
                    dataFloatVector.setValueCount(dataSize)
                    dataVector.setValueCount(dataSize)
                  case false =>
                }
              case MinorType.VARCHAR =>
                val varCharVector = dataDataVector.asInstanceOf[VarCharVector]
                val datas = data.asInstanceOf[ArrayBuffer[String]]
                val dataSize = datas.size
                for (j <- 0 until dataSize) {
                  dataVector.startNewValue(j)
                  val bytes = datas(j).getBytes
                  varCharVector.setIndexDefined(j)
                  varCharVector.setSafe(j, bytes)
                  dataVector.endValue(j, 1)
                }
                varCharVector.setValueCount(dataSize)
                dataVector.setValueCount(dataSize)
              case MinorType.VARBINARY =>
                val varBinaryVector = dataDataVector.asInstanceOf[VarBinaryVector]
                val datas = data.asInstanceOf[ArrayBuffer[String]]
                val dataSize = datas.size
                for (j <- 0 until dataSize) {
                  dataVector.startNewValue(j)
                  val bytes = datas(j).asInstanceOf[String].getBytes
                  varBinaryVector.setIndexDefined(j)
                  varBinaryVector.setValueLengthSafe(j, bytes.length)
                  varBinaryVector.setSafe(j, bytes)
                  dataVector.endValue(j, 1)
                }
                varBinaryVector.setValueCount(dataSize)
                dataVector.setValueCount(dataSize)
            }

            indicesShapeVector.allocateNew()
            val indicesShapeSize = indicesShape.size
            val indicesShapeIntVector = indicesShapeDataVector.asInstanceOf[IntVector]
            for (j <- 0 until indicesShapeSize) {
              indicesShapeVector.startNewValue(j)
              indicesShapeIntVector.setSafe(j, indicesShape(j))
              indicesShapeVector.endValue(j, 1)
            }
            indicesShapeIntVector.setValueCount(indicesShapeSize)
            indicesShapeVector.setValueCount(indicesShapeSize)

            indicesDataVector.allocateNew()
            val indicesDataIntVector = indicesDataDataVector.asInstanceOf[IntVector]
            val indicesDatas = indicesData.asInstanceOf[ArrayBuffer[Int]]
            val indicesDataSize = indicesDatas.size
            for (j <- 0 until indicesDataSize) {
              indicesDataVector.startNewValue(j)
              indicesDataIntVector.setSafe(j, indicesDatas(j))
              indicesDataVector.endValue(j, 1)
            }
            indicesDataIntVector.setValueCount(indicesDataSize)
            indicesDataVector.setValueCount(indicesDataSize)
          case _ =>
        }

      })
      arrowStreamWriter.writeBatch()
    }
    vectorSchemaRoot.close()
    arrowStreamWriter.end()
    arrowStreamWriter.close()
    byteArrayOutputStream.flush()
    byteArrayOutputStream.close()
    byteArrayOutputStream.toByteArray
  }

}

object Instances {
  def apply(instance: mutable.LinkedHashMap[String, Any]): Instances = {
    Instances(List(instance))
  }

  def apply(instances: mutable.LinkedHashMap[String, Any]*): Instances = {
    Instances(instances.toList)
  }

  def fromArrow(arrowBytes: Array[Byte]): Instances = {
    val instances = new mutable.ArrayBuffer[mutable.LinkedHashMap[String, Any]]()

    val byteArrayInputStream = new ByteArrayInputStream(arrowBytes)
    val rootAllocator = new RootAllocator(Integer.MAX_VALUE)
    val arrowStreamReader = new ArrowStreamReader(byteArrayInputStream, rootAllocator)
    val root = arrowStreamReader.getVectorSchemaRoot()
    val fieldVectors: util.List[FieldVector] = root.getFieldVectors

    while (arrowStreamReader.loadNextBatch()) {
      val map = new mutable.LinkedHashMap[String, Any]()
      fieldVectors.toArray().map(fieldVector => {
        val (name, value) =
          if (fieldVector.isInstanceOf[IntVector]) {
            val vector = fieldVector.asInstanceOf[IntVector]
            (vector.getName, vector.getObject(0))
          } else if (fieldVector.isInstanceOf[Float4Vector]) {
            val vector = fieldVector.asInstanceOf[Float4Vector]
            (vector.getName, vector.getObject(0))
          } else if (fieldVector.isInstanceOf[VarCharVector]) {
            val vector = fieldVector.asInstanceOf[VarCharVector]
            (vector.getName, new String(vector.getObject(0).getBytes))
          } else if (fieldVector.isInstanceOf[VarBinaryVector]) {
            val vector = fieldVector.asInstanceOf[VarBinaryVector]
            (vector.getName, new String(vector.getObject(0).asInstanceOf[Array[Byte]]))
          } else if (fieldVector.isInstanceOf[StructVector]) {
            val structVector = fieldVector.asInstanceOf[StructVector]
            val shapeVector = structVector.getChild("shape")
            val dataVector = structVector.getChild("data")
            val indicesShapeVector = structVector.getChild("indiceShape")
            val indicesDataVector = structVector.getChild("indiceData")
            val shapeDataVector = shapeVector.asInstanceOf[ListVector].getDataVector
            val dataDataVector = dataVector.asInstanceOf[ListVector].getDataVector
            val indicesShapeDataVector = indicesShapeVector.asInstanceOf[ListVector].getDataVector
            val indicesDataDataVector = indicesDataVector.asInstanceOf[ListVector].getDataVector

            val shape = new ArrayBuffer[Int]()
            for (i <- 0 until shapeDataVector.getValueCount) {
              shape.append(shapeDataVector.getObject(i).asInstanceOf[Int])
            }
            val data = dataDataVector.getMinorType match {
              case MinorType.INT =>
                val data = new ArrayBuffer[Float]()
                val dataIntVector = dataDataVector.asInstanceOf[IntVector]
                for (i <- 0 until dataIntVector.getValueCount) {
                  data.append(dataIntVector.getObject(i).toFloat)
                }
                data
              case MinorType.FLOAT4 =>
                val data = new ArrayBuffer[Float]()
                val dataFloatVector = dataDataVector.asInstanceOf[Float4Vector]
                for (i <- 0 until dataFloatVector.getValueCount) {
                  data.append(dataFloatVector.getObject(i).asInstanceOf[Float])
                }
                data
              case MinorType.VARCHAR =>
                val data = new ArrayBuffer[String]()
                val dataVarCharVector = dataDataVector.asInstanceOf[VarCharVector]
                for (i <- 0 until dataVarCharVector.getValueCount) {
                  data.append(
                    new String(dataVarCharVector.getObject(i).getBytes))
                }
                data
              case MinorType.VARBINARY =>
                val data = new ArrayBuffer[String]()
                val dataVarBinaryVector = dataDataVector.asInstanceOf[VarBinaryVector]
                for (i <- 0 until dataVarBinaryVector.getValueCount) {
                  data.append(
                    new String(dataVarBinaryVector.getObject(i).asInstanceOf[Array[Byte]]))
                }
                data
            }
            val indicesShape = new ArrayBuffer[Int]()
            for (i <- 0 until indicesShapeDataVector.getValueCount) {
              indicesShape.append(indicesShapeDataVector.getObject(i).asInstanceOf[Int])
            }
            val indicesData = new ArrayBuffer[Int]()
            for (i <- 0 until indicesDataDataVector.getValueCount) {
              indicesData.append(indicesDataDataVector.getObject(i).asInstanceOf[Int])
            }
            (structVector.getName, (shape, data, indicesShape, indicesData))
          } else {
            (null, null)
          }
        if (null != name) {
          map.put(name, value)
        }
      })
      instances.append(map)
    }

    arrowStreamReader.close()
    rootAllocator.close()
    byteArrayInputStream.close()
    new Instances(instances.toList)
  }

  def transferListToTensor(value: Any): (mutable.ArrayBuffer[Int], mutable.ArrayBuffer[Any]) = {
    val shape = mutable.ArrayBuffer[Int]()
    val data = mutable.ArrayBuffer[Any]()
    transferListToTensor(value, shape, data)
    val real = shape.take(shape.indexOf(-1))
    (real, data)
  }

  private def transferListToTensor(
                                    source: Any,
                                    shape: mutable.ArrayBuffer[Int],
                                    data: mutable.ArrayBuffer[Any]): Unit = {
    if (source.isInstanceOf[List[_]]) {
      val list = source.asInstanceOf[List[_]]
      shape.append(list.size)
      list.map(i => {
        transferListToTensor(i, shape, data)
      })
    } else {
      shape.append(-1)
      data.append(source)
    }
  }
}

case class Predictions[Type](predictions: Array[Type]) {
  override def toString: String = JsonUtil.toJson(this)
}

object Predictions {
  def apply[T](output: PredictionOutput[T])(implicit m: Manifest[T]): Predictions[T] = {
    Predictions(Array(output.result))
  }

  def apply[T](outputs: List[PredictionOutput[T]])(implicit m: Manifest[T]): Predictions[T] = {
    Predictions(outputs.map(_.result).toArray)
  }

  def apply[T](outputs: Seq[PredictionOutput[T]])(implicit m: Manifest[T]): Predictions[T] = {
    Predictions(outputs.map(_.result).toArray)
  }
}


case class ServingResponse[Type](statusCode: Int, entity: Type) {
  def this(tuple: (Int, Type)) = this(tuple._1, tuple._2)

  override def toString: String = s"[$statusCode, $entity]"

  def isSuccessful: Boolean = statusCode / 100 == 2
}

case class ServingRuntimeException(message: String = null, cause: Throwable = null)
  extends RuntimeException(message, cause) {
  def this(response: ServingResponse[String]) = this(JsonUtil.toJson(response), null)
}

case class ServingError(error: String) {
  override def toString: String = JsonUtil.toJson(this)
}

case class ServingTimerMetrics(
                                name: String,
                                count: Long,
                                meanRate: Double,
                                min: Long,
                                max: Long,
                                mean: Double,
                                median: Double,
                                stdDev: Double,
                                _75thPercentile: Double,
                                _95thPercentile: Double,
                                _98thPercentile: Double,
                                _99thPercentile: Double,
                                _999thPercentile: Double
                              )

object ServingTimerMetrics {
  def apply(name: String, timer: Timer): ServingTimerMetrics =
    ServingTimerMetrics(
      name,
      timer.getCount,
      timer.getMeanRate,
      timer.getSnapshot.getMin / 1000000,
      timer.getSnapshot.getMax / 1000000,
      timer.getSnapshot.getMean / 1000000,
      timer.getSnapshot.getMedian / 1000000,
      timer.getSnapshot.getStdDev / 1000000,
      timer.getSnapshot.get75thPercentile() / 1000000,
      timer.getSnapshot.get95thPercentile() / 1000000,
      timer.getSnapshot.get98thPercentile() / 1000000,
      timer.getSnapshot.get99thPercentile() / 1000000,
      timer.getSnapshot.get999thPercentile() / 1000000
    )
}


class ServableManager {
  private var modelVersionMap = new mutable.HashMap[String, mutable.HashMap[String, Servable]]

  def load(servableManagerConfigDir: String,
           purePredictTimersMap: mutable.HashMap[String, mutable.HashMap[String, Timer]],
           modelInferenceTimersMap: mutable.HashMap[String, mutable.HashMap[String, Timer]]):
                                                                                      Unit = {
    if (!Files.exists(Paths.get(servableManagerConfigDir))) {
      throw ServableLoadException("servable Manager config dir not exist", null)
    }
    val servableManagerConfYaml = scala.io.Source.fromFile(servableManagerConfigDir).mkString
    val modelInfoList = YamlUtil.fromYaml(classOf[ServableManagerConf],
      servableManagerConfYaml).modelMetaDataList
    for (modelInfo <- modelInfoList) {
      if (!modelVersionMap.contains(modelInfo.getModelName)) {
        val versionMapper = new mutable.HashMap[String, Servable]
        modelVersionMap(modelInfo.getModelName) = versionMapper
        val timerMapperInference = new mutable.HashMap[String, Timer]
        modelInferenceTimersMap(modelInfo.getModelName) = timerMapperInference
        val timerMapperPurePredict = new mutable.HashMap[String, Timer]
        purePredictTimersMap(modelInfo.getModelName) = timerMapperPurePredict
      }
      if (modelVersionMap(modelInfo.getModelName).contains(modelInfo.getModelVersion)) {
        throw ServableLoadException("duplicated model info. Model Name: " + modelInfo.getModelName
          + ", Model Version: " + modelInfo.getModelVersion, null)
      }
      modelInferenceTimersMap(modelInfo.getModelName)(modelInfo.getModelVersion) =
        metrics.timer("bigdl.serving.inference." + modelInfo.getModelName + "."
          + modelInfo.getModelVersion)
      purePredictTimersMap(modelInfo.getModelName)(modelInfo.getModelVersion) =
        metrics.timer("bigdl.pure.predict." + modelInfo.getModelName + "."
          + modelInfo.getModelVersion)
      val servable = modelInfo match {
        case clusterServingModelInfo: ClusterServingMetaData =>
          new ClusterServingServable(clusterServingModelInfo)
        case inferenceModelModelInfo: InferenceModelMetaData =>
          new InferenceModelServable(inferenceModelModelInfo,
            purePredictTimersMap(modelInfo.getModelName)(modelInfo.getModelVersion))
      }
      servable.load()
      modelVersionMap(modelInfo.getModelName)(modelInfo.getModelVersion) = servable
    }
  }

  def retriveAllServables: List[Servable] = {
    val result = modelVersionMap.values.flatMap(
      maps => maps.values.toList
    ).toList
    result
  }

  def retriveServables(modelName: String): List[Servable] = {
    if (!modelVersionMap.contains(modelName)) {
      throw ModelNotFoundException("model not exist. Model Name: " + modelName, null)
    }
    modelVersionMap(modelName).values.toList
  }

  def retriveServable(modelName: String, modelVersion: String): Servable = {
    if (!modelVersionMap.contains(modelName) || !modelVersionMap(modelName).contains
    (modelVersion)) {
      throw ModelNotFoundException("model not exist. Model Name: " + modelName +
        ", Model Version: " + modelVersion, null)
    }
    modelVersionMap(modelName)(modelVersion)
  }
}

abstract class Servable(modelMetaData: ModelMetaData) {
  def predict(input: Instances): Seq[PredictionOutput[String]]

  def predict(input: String): Seq[PredictionOutput[String]]

  def load(): Unit

  def getMetaData: ModelMetaData = modelMetaData
}


class InferenceModelServable(inferenceModelMetaData: InferenceModelMetaData,
                             purePredictTimer: Timer)
  extends Servable(inferenceModelMetaData) {
  val logger = LoggerFactory.getLogger(getClass)
  var model: InferenceModel = _
  var isFirstTimePredict = true

  def load(): Unit = {
    model = new InferenceModel(inferenceModelMetaData.modelConCurrentNum)
    inferenceModelMetaData.modelType match {
      case "OpenVINO" =>
        model.doLoadOpenVINO(inferenceModelMetaData.modelPath,
          inferenceModelMetaData.weightPath)
      case "tf.frozenModel" =>
        model.doLoadTensorflow(inferenceModelMetaData.modelPath, "frozenModel")
      case "BigDL" =>
        model.doLoadBigDL(inferenceModelMetaData.modelPath, inferenceModelMetaData.weightPath)
      case "Caffe" =>
        model.doLoadCaffe(inferenceModelMetaData.modelPath, inferenceModelMetaData.weightPath)
      case "PyTorch" =>
        model.doLoadPyTorch(inferenceModelMetaData.modelPath)
    }
    logger.info(s"model loaded successfully as $model")
  }

  def predict(inputs: Instances): Seq[PredictionOutput[String]] = {
    logger.info(s"inference model predict as instance")
    val activities = timing("activity make")(makeActivityTimer) {
      inputs.makeActivities(inferenceModelMetaData.features)
    }
    activities.map(
      activity => {
        val result = if (isFirstTimePredict) {
          timing("model first predict")() {
            isFirstTimePredict = false
            model.doPredict(activity)
          }
        } else {
          timing("model predict")(purePredictTimer) {
            model.doPredict(activity)
          }
        }
        timing("handle response")(handleResponseTimer) {
          val responses = tensorToString(result.toTensor[Float])
          PredictionOutput[String]("", responses)
        }
      })
  }

  def predict(input: String): Seq[PredictionOutput[String]] = {
    logger.info(s"inference model predict as string")
    val activities = timing("activity make")(makeActivityTimer) {
      JsonInputDomainDeser.deserialize(input)
    }
    activities.map(
      activity => {
        val result = if (isFirstTimePredict) {
          timing("model first predict")() {
            isFirstTimePredict = false
            model.doPredict(activity)
          }
        } else {
          timing("model predict")(purePredictTimer) {
            model.doPredict(activity)
          }
        }
        timing("handle response")(handleResponseTimer) {
          val responses = tensorToString(result.toTensor[Float])
          PredictionOutput[String]("", responses)
        }
      })
  }

  private def tensorToString(tensor: Tensor[Float]): String = {
    val outputShape = tensor.size()
    // Share Tensor Storage
    val jTensor = new com.intel.analytics.bigdl.orca.inference.JTensor(tensor.storage().array(),
      outputShape, false)
    """{ "data":""" + jTensor.getData.mkString(",") + """, "shape":""" +
      jTensor.getShape.mkString(",") + "}"
  }

}

class ClusterServingServable(clusterServingMetaData: ClusterServingMetaData)
  extends Servable(clusterServingMetaData) with Supportive {
  var redisPutter: ActorRef = _
  var redisGetter: ActorRef = _
  var querierQueue: LinkedBlockingQueue[ActorRef] = _
  val jedisPoolConfig = new JedisPoolConfig()
  jedisPoolConfig.setMaxTotal(256)
  val jedisPool = new JedisPool(jedisPoolConfig,
    clusterServingMetaData.redisHost, clusterServingMetaData.redisPort.toInt)
  val rateLimiter: RateLimiter = clusterServingMetaData.tokenBucketEnabled match {
    case true => RateLimiter.create(clusterServingMetaData.tokensPerSecond)
    case false => null
  }
  var ioActor: ActorRef = _


  def load(): Unit = {
    val redisPutterName = s"redis-putter-${clusterServingMetaData.modelName}" +
      s"-${clusterServingMetaData.modelVersion}"
    redisPutter = timing(s"$redisPutterName initialized.")() {
      val redisPutterProps = Props(new RedisPutActor(
        clusterServingMetaData.redisHost,
        clusterServingMetaData.redisPort.toInt,
        clusterServingMetaData.redisInputQueue,
        clusterServingMetaData.redisOutputQueue,
        clusterServingMetaData.timeWindow,
        clusterServingMetaData.countWindow,
        clusterServingMetaData.redisSecureEnabled,
        clusterServingMetaData.redisTrustStorePath,
        clusterServingMetaData.redisTrustStoreToken))
      system.actorOf(redisPutterProps, name = redisPutterName)
    }

    val redisGetterName = s"redis-getter-${clusterServingMetaData.modelName}" +
      s"-${clusterServingMetaData.modelVersion}"
    redisGetter = timing(s"$redisGetterName initialized.")() {
      val redisGetterProps = Props(new RedisGetActor(
        clusterServingMetaData.redisHost,
        clusterServingMetaData.redisPort.toInt,
        clusterServingMetaData.redisInputQueue,
        clusterServingMetaData.redisOutputQueue,
        clusterServingMetaData.redisSecureEnabled,
        clusterServingMetaData.redisTrustStorePath,
        clusterServingMetaData.redisTrustStoreToken))
      system.actorOf(redisGetterProps, name = redisGetterName)
    }

    val querierNum = 1000
    querierQueue = timing(s"queriers initialized.")() {
      val querierQueue = new LinkedBlockingQueue[ActorRef](querierNum)
      val querierProps = Props(new QueryActor(redisGetter))
      List.range(0, querierNum).map(index => {
        val querierName = s"querier-$index-${clusterServingMetaData.modelName}" +
          s"-${clusterServingMetaData.modelVersion}"
        val querier = system.actorOf(querierProps, name = querierName)
        querierQueue.put(querier)
      })
      querierQueue
    }
    val actorName = s"redis-ioActor-getter-${clusterServingMetaData.modelName}" +
      s"-${clusterServingMetaData.modelVersion}"
    ioActor = timing(s"$actorName initialized.")() {
      val getterProps = Props(new RedisIOActor(jedisPool = jedisPool,
        redisOutputQueue = clusterServingMetaData.redisOutputQueue,
        redisInputQueue = clusterServingMetaData.redisInputQueue))
      system.actorOf(getterProps, name = actorName)
    }
    system.scheduler.schedule(1 milliseconds, 1 millisecond,
      ioActor, DequeueMessage())(system.dispatcher)
  }

  def predict(input: String): Seq[PredictionOutput[String]] = {
    val id = UUID.randomUUID().toString
    val result = timing("response waiting")() {
      val id = UUID.randomUUID().toString
      val results = timing(s"query message wait for key $id")() {
        Await.result(ioActor ? DataInputMessage(id, input), timeout.duration)
          .asInstanceOf[ModelOutputMessage].valueMap
      }
      val objectMapper = new ObjectMapper()
      results.map(r => {
        val resultStr = objectMapper.writeValueAsString(r._2)
        PredictionOutput(r._1, resultStr)
      })
    }
    result.toSeq
  }

  def predict(instances: Instances):
  Seq[PredictionOutput[String]] = {
    logger.info(s"cluster serving predict as instance")
    val inputs = instances.instances.map(instance => {
      InstancesPredictionInput(Instances(instance))
    })
    timing("put message send")() {
      val message = PredictionInputMessage(inputs)
      redisPutter ! message
    }
    val result = timing("response waiting")() {
      val ids = inputs.map(_.getId())
      val queryMessage = PredictionQueryMessage(ids)
      val querier = timing("querier take")() {
        querierQueue.take()
      }
      val results = timing(s"query message wait for key $ids")(
        overallRequestTimer, waitRedisTimer) {
        Await.result(querier ? queryMessage, timeout.duration)
          .asInstanceOf[Seq[(String, util.Map[String, String])]]
      }
      timing("querier back")() {
        querierQueue.offer(querier)
      }
      val objectMapper = new ObjectMapper()
      results.map(r => {
        val resultStr = objectMapper.writeValueAsString(r._2)
        PredictionOutput(r._1, resultStr)
      })
    }
    result
  }

  override def getMetaData: ModelMetaData = {
    ClusterServingMetaData(clusterServingMetaData.modelName, clusterServingMetaData.modelVersion,
      clusterServingMetaData.redisHost, clusterServingMetaData.redisPort,
      clusterServingMetaData.redisInputQueue, clusterServingMetaData.redisOutputQueue,
      clusterServingMetaData.timeWindow, clusterServingMetaData.countWindow,
      clusterServingMetaData.tokenBucketEnabled, clusterServingMetaData.tokensPerSecond,
      clusterServingMetaData.redisSecureEnabled, "*******", "*******",
      clusterServingMetaData.inputCompileType, clusterServingMetaData.features)
  }

}


@JsonTypeInfo(
  use = JsonTypeInfo.Id.NAME,
  include = JsonTypeInfo.As.PROPERTY,
  property = "type"
)
@JsonSubTypes(Array(
  new Type(value = classOf[InferenceModelMetaData], name = "InferenceModelMetaData"),
  new Type(value = classOf[ClusterServingMetaData], name = "ClusterServingMetaData")
))
abstract class ModelMetaData(modelName: String, modelVersion: String, features: Array[String]) {
  def getModelName: String = {
    modelName
  }

  def getModelVersion: String = {
    modelVersion
  }
}

case class InferenceModelMetaData(modelName: String,
                                  modelVersion: String,
                                  modelPath: String = "",
                                  modelType: String = "",
                                  weightPath: String = "",
                                  modelConCurrentNum: Int = 1,
                                  inputCompileType: String = "direct", // direct or instance
                                  features: Array[String])
  extends ModelMetaData(modelName, modelVersion, features)

case class ClusterServingMetaData(modelName: String,
                                  modelVersion: String,
                                  redisHost: String,
                                  redisPort: String,
                                  redisInputQueue: String,
                                  redisOutputQueue: String,
                                  timeWindow: Int = 0,
                                  countWindow: Int = 56,
                                  tokenBucketEnabled: Boolean = false,
                                  tokensPerSecond: Int = 100,
                                  redisSecureEnabled: Boolean = false,
                                  redisTrustStorePath: String = null,
                                  redisTrustStoreToken: String = "1234qwer",
                                  inputCompileType: String = "direct",
                                  features: Array[String])
  extends ModelMetaData(modelName, modelVersion, features)


case class ServableManagerConf(modelMetaDataList: List[ModelMetaData])

case class ModelNotFoundException(message: String = null, cause: Throwable = null)
  extends RuntimeException(message, cause) {
  def this(response: ServingResponse[String]) = this(JsonUtil.toJson(response), null)
}

case class ServableLoadException(message: String = null, cause: Throwable = null)
  extends RuntimeException(message, cause) {
  def this(response: ServingResponse[String]) = this(JsonUtil.toJson(response), null)
}


class JsonInputDomainDeser extends JsonDeserializer[Seq[Activity]]{
  var intBuffer: ArrayBuffer[Int] = null
  var floatBuffer: ArrayBuffer[Float] = null
  var stringBuffer: ArrayBuffer[String] = null
  var shapeBuffer: ArrayBuffer[Int] = null
  var valueCount: Int = 0
  var shapeMask: Map[Int, Boolean] = null
  override def deserialize(p: JsonParser, ctxt: DeserializationContext): Seq[Activity] = {
    val oc = p.getCodec
    val node = oc.readTree[JsonNode](p)
    (1 to node.get("instances").size()).map(i => {
      val inputsIt = node.get("instances").get(i-1).elements()
      val tensorBuffer = new ArrayBuffer[Tensor[Float]]()
      while (inputsIt.hasNext) {
        initBuffer()
        parse(inputsIt.next(), 0)
        if (shapeBuffer.isEmpty) shapeBuffer.append(1)
        if (!floatBuffer.isEmpty) {
          tensorBuffer.append(Tensor[Float](floatBuffer.toArray, shapeBuffer.toArray))
        } else {
          // add string, string tensor, sparse tensor in the future
          throw new Error("???")
        }

      }
      T.array(tensorBuffer.toArray)
    })
  }
  def parse(node: JsonNode, currentShapeDim: Int): Unit = {
    if (node.isInstanceOf[ArrayNode]) {

      val iter = node.elements()
      if (shapeMask.get(currentShapeDim) == None) {
        shapeBuffer.append(node.size())
        shapeMask += (currentShapeDim -> true)
      }
      while (iter.hasNext) {
        parse(iter.next(), currentShapeDim + 1)
      }
    } else if (node.isInstanceOf[TextNode]) {
      stringBuffer.append(node.asText())
    } else if (node.isInstanceOf[ObjectNode]) {
      // currently used for SparseTensor only maybe
    } else {
      // v1: int, float, double would all parse to float
      floatBuffer.append(node.asDouble().toFloat)
    }

  }
  def initBuffer(): Unit = {
    floatBuffer = new ArrayBuffer[Float]()
    shapeBuffer = new ArrayBuffer[Int]()
    stringBuffer = new ArrayBuffer[String]()
    shapeMask = Map[Int, Boolean]()
  }

}

object JsonInputDomainDeser {
  def deserialize(str: String): Seq[Activity] = {
    val mapper = new ObjectMapper()
    val module = new SimpleModule()
    module.addDeserializer(classOf[Seq[Activity]], new JsonInputDomainDeser())
    mapper.registerModule(module)
    mapper.readValue(str, classOf[Seq[Activity]])
  }
}
