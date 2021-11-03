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

package com.intel.analytics.bigdl.serving.benchmark

import com.codahale.metrics.MetricRegistry
import com.intel.analytics.bigdl.orca.inference.InferenceModel
import com.intel.analytics.bigdl.serving.http.{JsonUtil, ServingTimerMetrics, Supportive}
import com.intel.analytics.bigdl.serving.serialization.JsonInputDeserializer
import scopt.OptionParser

object TestUtils {
  def getStrFromResourceFile(path: String): String = {
    scala.io.Source.fromFile(path).mkString
  }
}

object Operations extends Supportive {
  // initialize the parser
  case class Config(modelPath: String = null, jsonPath: String = null)
  val parser = new OptionParser[Config]("DIEN benchmark test Usage") {
    opt[String]('m', "modelPath")
      .text("Model Path for Test")
      .action((x, params) => params.copy(modelPath = x))
      .required()
    opt[String]('j', "jsonPath")
      .text("Json Format Input Path of Model")
      .action((x, params) => params.copy(jsonPath = x))
      .required()
  }

  // val logger = Logger.getLogger(getClass)
  def main(args: Array[String]): Unit = {
    // read the path of model
    val arg = parser.parse(args, Config()).head
    val path = arg.modelPath
    val jsonPath = arg.jsonPath

    // read file from path to String
    // this is a prepared json format input of DIEN recommendation model
    val string = TestUtils.getStrFromResourceFile(jsonPath)

    // decode json string input to activity
    val input = JsonInputDeserializer.deserialize(string)


    (1 to 4).foreach(threadNumber => {
      // load model with concurrent number 1~4
      val model = new InferenceModel(threadNumber)
      model.doLoadTensorflow(path, "frozenModel")

      (0 to 10).foreach(range => {
        logger.info(s"inference with $threadNumber threads and range $range starts.\n")
        // set timer name
        val preprocessingKey = s"preprocessing.${threadNumber}_thread.${range}_range"
        val postprocessingKey = s"postprocessing.${threadNumber}_thread.${range}_range"
        val predictKey = s"predict.${threadNumber}_thread.${range}_range"
        // initialize timers
        val preprocessingTimer = metrics.timer(preprocessingKey)
        val postprocessingTimer = metrics.timer(postprocessingKey)
        val predictTimer = metrics.timer(predictKey)

        (0 until threadNumber).indices.toParArray.foreach(threadIndex => {

          (0 until 100).foreach(iter => {
            // do the mock operation 0 to 10 times to mock preprocessing
            timing("preprocessing")(preprocessingTimer) {
              (0 until range).foreach(iter => {
                mockOperation1ms()
              })
            }

            // do predict
            timing(s"thread $threadIndex predict")(predictTimer) {
              val result = model.doPredict(input)
            }

            // do the mock operation 0 to 10 times to mock postprocessing
            timing("postprocessing")(postprocessingTimer) {
              (0 until range).foreach(iter => {
                mockOperation1ms()
              })
            }
            // sleep 0 to 10 ms
            Thread.sleep(range)
          })
        })
        // output metrics
        val servingMetricsList = List(
          ServingTimerMetrics(preprocessingKey, preprocessingTimer),
          ServingTimerMetrics(predictKey, predictTimer),
          ServingTimerMetrics(postprocessingKey, postprocessingTimer))
        val jsonMetrics = JsonUtil.toJson(servingMetricsList)
        logger.info(jsonMetrics)
      })
    })
  }

  // This function will take around 1ms to run
  // Run different times of the function to mock different
  // pre- and post-processing time
  def mockOperation1ms() : Unit = {
    var num = 0
    for (i <- 0 to 200000) {
      num += 1
    }
  }

  val metrics = new MetricRegistry

}
