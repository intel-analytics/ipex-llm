package com.intel.analytics.zoo.apps.model.inference.flink.Resnet50ImageClassification

import java.io.{File, FileInputStream}
import java.util.{Arrays, List => JList}

import com.intel.analytics.zoo.pipeline.inference.JTensor
import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.api.java.utils.ParameterTool
import org.apache.flink.configuration.Configuration
import org.apache.flink.streaming.api.datastream.DataStreamUtils
import org.apache.flink.streaming.api.scala.{StreamExecutionEnvironment, _}

import scala.collection.JavaConverters._

object ImageClassificationStreaming {

  def main(args: Array[String]): Unit = {
    var modelType = "resnet_v1_50"
    var checkpointPathcheckpointPath: String = "/path/to/models/resnet_v1_50.ckpt"
    var ifReverseInputChannels = true
    var inputShape = Array(1, 224, 224, 3)
    var meanValues = Array(123.68f, 116.78f, 103.94f)
    var scale = 1.0f

    try {
      val params = ParameterTool.fromArgs(args)
      modelType = params.get("modelType")
      checkpointPathcheckpointPath = params.get("checkpointPathcheckpointPath")
      inputShape = if (params.has("inputShape")) {
        val inputShapeStr = params.get("inputShape")
        inputShapeStr.split(",").map(_.toInt).toArray
      } else Array(1, 224, 224, 3)
      ifReverseInputChannels = if (params.has("ifReverseInputChannels")) params.getBoolean("ifReverseInputChannels") else true
      meanValues = if (params.has("meanValues")) {
        val meanValuesStr = params.get("meanValues")
        meanValuesStr.split(",").map(_.toFloat).toArray
      } else Array(123.68f, 116.78f, 103.94f)
      scale = if (params.has("scale")) params.getFloat("scale") else 1.0f
    } catch {
      case e: Exception => {
        System.err.println("Please run 'ImageClassificationStreaming --modelType <modelType> --checkpointPathcheckpointPath <checkpointPathcheckpointPath> " +
          "--inputShape <inputShapes> --ifReverseInputChannels <ifReverseInputChannels> --meanValues <meanValues> --scale <scale>" +
          "--parallelism <parallelism>'.")
        return
      }
    }

    println("start ImageClassificationStreaming job...")
    println("params resolved", modelType, checkpointPathcheckpointPath, inputShape.mkString(","), ifReverseInputChannels, meanValues.mkString(","), scale)

    val classLoader = this.getClass.getClassLoader
    val content = classLoader.getResourceAsStream("n02110063_11239.JPEG")
    val imageBytes = Stream.continually(content.read).takeWhile(_ != -1).map(_.toByte).toArray
    val imageProcess = new ImageProcesser(imageBytes, 224, 224)
    val res = imageProcess.preProcess(imageBytes, 224, 224)
    val input = new Array[Float](res.nElement())
    val inputs = List.fill(100)(input)

    val fileSize = new File(checkpointPathcheckpointPath).length()
    val inputStream = new FileInputStream(checkpointPathcheckpointPath)
    val modelBytes = new Array[Byte](fileSize.toInt)
    inputStream.read(modelBytes)

    val env: StreamExecutionEnvironment = StreamExecutionEnvironment.getExecutionEnvironment
    println(env.getConfig)

    val dataStream: DataStream[Array[Float]] = env.fromCollection(inputs)
    val tensorStream: DataStream[JList[JList[JTensor]]] = dataStream.map(value => {
      val input = new JTensor(value, Array(1, 224, 224, 3))
      val data = Arrays.asList(input)
      List(data).asJava
    })

    val resultStream = tensorStream.map(new ModelPredictionMapFunction(modelType, modelBytes, inputShape, ifReverseInputChannels, meanValues, scale))

    env.execute("ImageClassificationStreaming")

    val results = DataStreamUtils.collect(resultStream.javaStream).asScala

    println(" Printing result to stdout.")
    results.foreach(println)
  }
}

class ModelPredictionMapFunction(modelType: String, modelBytes: Array[Byte], inputShape: Array[Int], ifReverseInputChannels: Boolean, meanValues: Array[Float], scale: Float) extends RichMapFunction[JList[JList[JTensor]], JList[JList[JTensor]]] {
  var resnet50InferenceModel: Resnet50InferenceModel = _
  override def open(parameters: Configuration): Unit = {
    resnet50InferenceModel = new Resnet50InferenceModel(1, modelType, modelBytes, inputShape, ifReverseInputChannels, meanValues, scale)
  }
  override def close(): Unit = {
    resnet50InferenceModel.doRelease()
  }
  override def map(in: JList[JList[JTensor]]): JList[JList[JTensor]] = {
    resnet50InferenceModel.doPredict(in)
  }
}
