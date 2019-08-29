package com.intel.analytics.zoo.apps.model.inference.flink.TextClassification

import scala.collection.JavaConverters._
import java.util.{List => JList}
import java.util

import com.intel.analytics.zoo.pipeline.inference.JTensor
import org.apache.flink.api.java.utils.ParameterTool
import org.apache.flink.streaming.api.scala._
import org.apache.flink.streaming.api.windowing.time.Time

object TextClassificationStreaming {

  def main(args: Array[String]) : Unit = {
    // the host and the port to connect to
    var hostname: String = "localhost"
    var port: Int = 0
    var supportedConcurrentNum = 1
    var stopWordsCount = 10
    var sequenceLength = 200
    var embeddingFilePath = "./glove.6B.300d.txt"
    var modelPath = "./text-classification.bigdl"
    var parallelism = 2

    try {
      val params = ParameterTool.fromArgs(args)
      hostname = if (params.has("hostname")) params.get("hostname") else "localhost"
      port = params.getInt("port")
      supportedConcurrentNum = if(params.has("supportedConcurrentNum")) params.getInt("supportedConcurrentNum") else 1
      stopWordsCount = if(params.has("stopWordsCount")) params.getInt("stopWordsCount") else 10
      sequenceLength = if(params.has("sequenceLength")) params.getInt("sequenceLength") else 200
      embeddingFilePath = if(params.has("embeddingFilePath")) params.get("embeddingFilePath") else "./glove.6B.300d.txt"
      modelPath = if(params.has("modelPath")) params.get("modelPath") else "./text-classification.bigdl"
      parallelism = if(params.has("parallelism")) params.getInt("parallelism") else 2
    } catch {
      case e: Exception => {
        System.err.println("Please run 'TextClassificationStreaming --hostname <hostname> --port <port> " +
          "--supportedConcurrentNum <supportedConcurrentNum> --stopWordsCount <stopWordsCount> --sequenceLength <sequenceLength>" +
          "--embeddingFilePath <embeddingFilePath> --modelPath <modelPath> --parallelism <parallelism>'.")
        System.err.println("To start a simple text server, run 'netcat -l <port>' and type the input text into the command line")
        return
      }
    }

    println("start TextClassificationStreaming job...")

    val env: StreamExecutionEnvironment = StreamExecutionEnvironment.getExecutionEnvironment

    val model = new TextClassificationInferenceModel(supportedConcurrentNum, stopWordsCount, sequenceLength, embeddingFilePath)
    model.doLoad(modelPath)
    println(model)
    val inputTensor: JTensor = model.preprocess("hello world")
    val input: JList[JTensor] = List(inputTensor).asJava
    val inputs = new util.ArrayList[JList[JTensor]]()
    inputs.add(input)
    println("################" + model.doPredict(inputs))

    val textStream: DataStream[String] = env.socketTextStream(hostname, port, '\n')//.timeWindow(Time.seconds(5))

    textStream.map(text => {
      val inputTensor: JTensor = model.preprocess(text)
      val input: JList[JTensor] = List(inputTensor).asJava
      val inputs = new util.ArrayList[JList[JTensor]]()
      inputs.add(input)
      val result = model.doPredict(inputs)
      println(s"$text $model")
      println("*******************", text, result)
    }).setParallelism(2)

    env.execute("Socket Window TextClassificationStreaming")
  }

}
