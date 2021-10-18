package com.intel.analytics.zoo.apps.model.inference.flink.ImageClassification

import java.io.{File, FileInputStream}
import java.{io, util}
import java.util.{List => JList}

import com.intel.analytics.zoo.pipeline.inference.JTensor
import org.apache.commons.io.FileUtils
import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.api.java.utils.ParameterTool
import org.apache.flink.configuration.Configuration
import org.apache.flink.streaming.api.datastream
import org.apache.flink.streaming.api.datastream.{DataStreamSink, DataStreamUtils}
import org.apache.flink.streaming.api.scala.{StreamExecutionEnvironment, _}

import scala.collection.JavaConverters._
import scala.io.Source

object ImageClassificationStreaming {

  def main(args: Array[String]): Unit = {

    // Define parameters
    // Define and obtain arguments from Params
    var modelPath = "/path/to/modelFile"
    var modelType = "frozenModel"
    var modelInputs = Array("input:0")
    var modelOutputs = Array("MobilenetV1/Predictions/Reshape_1:0")
    var imageDir = "/path/to/imageDir"
    var classesFile = "/path/to/labelFile"
    var intraOpParallelismThreads = 1
    var interOpParallelismThreads = 1
    var usePerSessionThreads = true
    var output = "/path/to/output"
    val params = ParameterTool.fromArgs(args)

    try {
      modelPath = params.get("modelPath")
      modelType = params.get("modelType")
      imageDir = params.get("images")
      classesFile = params.get("classes")
      modelInputs = if (params.has("modelInputs")) Array(params.get("modelInputs")) else Array("input:0")
      modelOutputs = if (params.has("modelOutputs")) Array(params.get("modelOutputs")) else Array("MobilenetV1/Predictions/Reshape_1:0")
      intraOpParallelismThreads = if (params.has("intraOpParallelismThreads")) params.getInt("intraOpParallelismThreads") else 1
      interOpParallelismThreads = if (params.has("interOpParallelismThreads")) params.getInt("interOpParallelismThreads") else 1
      usePerSessionThreads = if (params.has("usePerSessionThreads")) params.getBoolean("usePerSessionThreads") else true
    } catch {
      case e: Exception => {
        System.err.println("Please run ImageClassificationStreaming --modelPath <modelPath> --modelType <modelType> " +
          "--images <imageDir> --classes <classesFile> --modelInputs <modelInputs> --modelOutputs <modelOutputs> --intraOpParallelismThreads <intraOpParallelismThreads> --interOpParallelismThreads <interOpParallelismThreads> --usePerSessionThreads <usePerSessionThreads>" +
          "--parallelism <parallelism>'.")
        return
      }
    }

    println("start ImageClassificationStreaming job...")

    // ImageNet labels
    val labels = Source.fromFile(classesFile).getLines.toList

    // Image loading and pre-processing
    // Load images from folder, and hold images as a list
    val fileList = new File(imageDir).listFiles.toList
    println("ImageList", fileList)

    // Image pre-processing
    val inputImages = fileList.map(file => {
      // Read image as Array[Byte]
      val imageBytes = FileUtils.readFileToByteArray(file)
      // Execute image processing with ImageProcessor class
      val imageProcess = new ImageProcessor
      val res = imageProcess.preProcess(imageBytes, 224, 224)
      // Convert input to List[List[JTensor]]]
      val input = new JTensor(res, Array(1, 224, 224, 3))
      List(util.Arrays.asList(input)).asJava
    })

    // Getting started the Flink Program
    // Obtain a Flink execution environment
    val env: StreamExecutionEnvironment = StreamExecutionEnvironment.getExecutionEnvironment

    // Create and transform DataStreams
    val dataStream: DataStream[JList[JList[JTensor]]] = env.fromCollection(inputImages)

    // Specify the transformation functions
    // Before that, define class to extend InferenceModel to load the pre-trained model. And specify the map fucntion with InferenceModel predict function.
    val resultStream = dataStream.map(new ModelPredictionMapFunction(modelPath, modelType, modelInputs, modelOutputs, intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads))

    // Obtain classfication label by index
    val results = resultStream.map(i => labels(i - 1))

    // Print results to file or stdout
    if (params.has("output")) {
      results.writeAsText(params.get("output")).setParallelism(1)
    } else {
      println("Printing result to stdout. Use --output to specify output path.");
      results.print()
    }

    // Trigger the program execution on Flink
    env.execute("ImageClassificationStreaming")
  }

}

class ModelPredictionMapFunction(modelPath: String, modelType: String, inputs: Array[String], outputs: Array[String], intraOpParallelismThreads: Int, interOpParallelismThreads: Int, usePerSessionThreads: Boolean) extends RichMapFunction[JList[JList[JTensor]], Int] {
  var MobileNetInferenceModel: MobileNetInferenceModel = _

  override def open(parameters: Configuration): Unit = {
    MobileNetInferenceModel = new MobileNetInferenceModel(1, modelPath, modelType, inputs, outputs, intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads)
  }

  override def close(): Unit = {
    MobileNetInferenceModel.doRelease()
  }

  override def map(in: JList[JList[JTensor]]): (Int) = {
    val outputData = MobileNetInferenceModel.doPredict(in).get(0).get(0).getData
    val max: Float = outputData.max
    val index = outputData.indexOf(max)
    (index)
  }
}
