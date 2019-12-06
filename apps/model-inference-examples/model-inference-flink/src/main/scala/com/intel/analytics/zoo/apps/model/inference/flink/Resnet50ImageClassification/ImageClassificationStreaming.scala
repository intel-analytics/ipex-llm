package com.intel.analytics.zoo.apps.model.inference.flink.Resnet50ImageClassification

import java.io.{File, FileInputStream}
import java.util
import java.util.{List => JList}

import com.intel.analytics.zoo.pipeline.inference.JTensor
import org.apache.commons.io.FileUtils
import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.api.java.utils.ParameterTool
import org.apache.flink.configuration.Configuration
import org.apache.flink.streaming.api.datastream.DataStreamUtils
import org.apache.flink.streaming.api.scala.{StreamExecutionEnvironment, _}

import scala.collection.JavaConverters._
import scala.io.Source

object ImageClassificationStreaming {

  def main(args: Array[String]): Unit = {

    // Define parameters
    // Define and obtain arguments from Params
    var modelType = "resnet_v1_50"
    var checkpointPath: String = "/path/to/checkpointFile"
    var ifReverseInputChannels = true
    var inputShape = Array(1, 224, 224, 3)
    var imageDir = "/path/to/imageDir"
    var classesFile = "/path/to/labelFile"
    var meanValues = Array(123.68f, 116.78f, 103.94f)
    var scale = 1.0f

    try {
      val params = ParameterTool.fromArgs(args)
      modelType = params.get("modelType")
      checkpointPath = params.get("checkpointPath")
      imageDir = params.get("image")
      classesFile = params.get("classes")
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
        System.err.println("Please run 'ImageClassificationStreaming --modelType <modelType> --checkpointPath <checkpointPath> " +
          "--inputShape <inputShapes> --ifReverseInputChannels <ifReverseInputChannels> --imageDir <imageDir> --classesFile <classesFile> --meanValues <meanValues> --scale <scale>" +
          "--parallelism <parallelism>'.")
        return
      }
    }

    println("params resolved", modelType, checkpointPath, imageDir, classesFile, inputShape.mkString(","), ifReverseInputChannels, meanValues.mkString(","), scale)
    println("start ImageClassificationStreaming job...")

    // Define modelBytes
    val fileSize = new File(checkpointPath).length()
    val inputStream = new FileInputStream(checkpointPath)
    val modelBytes = new Array[Byte](fileSize.toInt)
    inputStream.read(modelBytes)

    // Image loading and pre-processing
    // Load images from folder, and hold images as a list
    val fileList = new File(imageDir).listFiles.toList
    println("ImageList", fileList)

    // Image pre-processing
    val inputs = fileList.map(file => {
      val imageBytes = FileUtils.readFileToByteArray(file)
      val imageProcess = new ImageProcessor
      val res = imageProcess.preProcess(imageBytes, 224, 224, 123, 116, 103, 1.0)
      val input = new JTensor(res, Array(1, 224, 224, 3))
      List(util.Arrays.asList(input)).asJava
    })

    // Getting started the Flink Program
    // Obtain a Flink execution environment
    val env: StreamExecutionEnvironment = StreamExecutionEnvironment.getExecutionEnvironment

    // Create and transform DataStreams
    val dataStream: DataStream[JList[JList[JTensor]]] = env.fromCollection(inputs)

    // Specify the transformation functions
    // First define an Analytics Zoo InferenceModel class to load the pre-trained model. And specify the map fucntion with InferenceModel predict function.
    val resultStream = dataStream.map(new ModelPredictionMapFunction(modelType, modelBytes, inputShape, ifReverseInputChannels, meanValues, scale))

    // Trigger the program execution on Flink
    env.execute("ImageClassificationStreaming")

    // Collect final results, and print predicted classes
    val results = DataStreamUtils.collect(resultStream.javaStream).asScala

    println("Printing result ...")
    val labels = Source.fromFile(classesFile).getLines.toList
    results.foreach((i) => println(labels(i)))
  }

}

class ModelPredictionMapFunction(modelType: String, modelBytes: Array[Byte], inputShape: Array[Int], ifReverseInputChannels: Boolean, meanValues: Array[Float], scale: Float) extends RichMapFunction[JList[JList[JTensor]], Int] {
  var resnet50InferenceModel: Resnet50InferenceModel = _

  override def open(parameters: Configuration): Unit = {
    resnet50InferenceModel = new Resnet50InferenceModel(1, modelType, modelBytes, inputShape, ifReverseInputChannels, meanValues, scale)
  }

  override def close(): Unit = {
    resnet50InferenceModel.doRelease()
  }

  override def map(in: JList[JList[JTensor]]): (Int) = {
    val outputData = resnet50InferenceModel.doPredict(in).get(0).get(0).getData
    val max: Float = outputData.max
    val index = outputData.indexOf(max)
    (index)
  }
}
