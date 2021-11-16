package com.intel.analytics.bigdl.ppml.vfl.utils

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.keras.models.InternalOptimizerUtil.getParametersFromModel
import com.intel.analytics.bigdl.ppml.FLClient
import com.intel.analytics.bigdl.ppml.generated.FLProto
import com.intel.analytics.bigdl.ppml.generated.FLProto.{DownloadResponse, FloatTensor, Table, TableMetaData}
import org.apache.log4j.Logger

import scala.reflect.ClassTag
import scala.util.Random
import scala.collection.JavaConversions._
import scala.collection.JavaConverters._


object ProtoUtils {
  private val logger = Logger.getLogger(getClass)
  def outputTargetToTableProto(output: Activity,
                               target: Activity,
                               meta: TableMetaData = null): FLProto.Table = {
    val tensorProto = toFloatTensor(output.toTensor[Float])

    val builder = Table.newBuilder
      .putTable("output", tensorProto)
    if (meta != null) {
      builder.setMetaData(meta)
    }

    if (target != null) {
      val targetTensor = toFloatTensor(target.toTensor[Float])
      builder.putTable("target", targetTensor)
    }
    builder.build()
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
  def uploadModel(client: FLClient, model: Module[Float], flVersion: Int): Unit = {
    val parameterTable = model.getParametersTable()

    val metadata = TableMetaData.newBuilder
      .setName("test").setVersion(flVersion).build

    val weights = getParametersFromModel(model)._1

    val tensor =
      FloatTensor.newBuilder()
        .addAllTensor(weights.storage.toList.map(v => float2Float(v)))
        .addAllShape(weights.size.toList.map(v => int2Integer(v)))
        .build()


    val metamodel = FLProto.Table.newBuilder
      .putTable("weights", tensor)
      .setMetaData(metadata)
      .build
    client.nnStub.uploadTrain(metamodel)
  }

  def updateModel(model: Module[Float],
                  modelData: FLProto.Table): Unit = {
    val weigthBias = modelData.getTableMap.get("weights")
    val data = weigthBias.getTensorList.asScala.map(v => Float2float(v)).toArray
    val shape = weigthBias.getShapeList.asScala.map(v => Integer2int(v)).toArray
    val tensor = Tensor(data, shape)
    getParametersFromModel(model)._1.copy(tensor)
  }

  def getTensor(name: String, modelData: FLProto.Table): Tensor[Float] = {
    val dataMap = modelData.getTableMap.get(name)
    val data = dataMap.getTensorList.asScala.map(Float2float).toArray
    val shape = dataMap.getShapeList.asScala.map(Integer2int).toArray
    Tensor[Float](data, shape)
  }


  def downloadTrain(client: FLClient, modelName: String, flVersion: Int): FLProto.Table = {
    var code = 0
    var maxRetry = 20000
    var downloadRes = None: Option[DownloadResponse]
    while (code == 0 && maxRetry > 0) {
      downloadRes = Some(client.nnStub.downloadTrain(modelName, flVersion))
      code = downloadRes.get.getCode
      if (code == 0) {
        logger.info("Waiting 10ms for other clients!")
        maxRetry -= 1
        Thread.sleep(10)
      }
    }
    if (code == 0) throw new Exception("Failed to download model within max retries!")
    downloadRes.get.getData
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

  def almostEqual(v1: Float, v2: Float): Boolean = {
    if (math.abs(v1 - v2) <= 1e-1f)
      true
    else
      false
  }




}
