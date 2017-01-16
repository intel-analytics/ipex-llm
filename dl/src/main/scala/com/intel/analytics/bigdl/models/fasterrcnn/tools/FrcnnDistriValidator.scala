/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.models.fasterrcnn.tools

import com.intel.analytics.bigdl.models.fasterrcnn.dataset.Target
import com.intel.analytics.bigdl.models.fasterrcnn.dataset.transformers.MiniBatch
import com.intel.analytics.bigdl.models.fasterrcnn.model.FasterRcnn
import com.intel.analytics.bigdl.models.fasterrcnn.tools.FrcnnDistriValidator._
import com.intel.analytics.bigdl.models.fasterrcnn.utils.PascalVocEvaluator
import com.intel.analytics.bigdl.models.utils.ModelBroadCast
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.Logger
import org.apache.spark.rdd.RDD

object FrcnnDistriValidator {
  val logger = Logger.getLogger(this.getClass)
}

class FrcnnDistriValidator(net: FasterRcnn, classNum: Int,
  rdd: RDD[MiniBatch], maxPerImage: Int = 100, thresh: Double = 0.05) {
  def assertEngineInited(): Unit = {
    require(Engine.isInitialized, s"you may forget to initialize Engine.")
  }

  val model = net.getTestModel
  val predParam = PredictorParam(net.param.NMS, classNum, net.param.BBOX_VOTE,
    maxPerImage, thresh)

  def test(evaluator: Option[PascalVocEvaluator] = None):
  Array[(Array[Target], Tensor[Float], Tensor[Float], String)] = {
    this.assertEngineInited()
    val predictor = new Predictor(predParam)
    val broadcastModel = ModelBroadCast().broadcast(rdd.sparkContext, model)
    val broadcastPredictor = rdd.sparkContext.broadcast(predictor)
    val broadcastEvaluator = rdd.sparkContext.broadcast(evaluator)

    logger.info(s"partition number is set to ${ rdd.partitions.length }")
    val start = System.nanoTime()
    val output = rdd.mapPartitions(dataIter => {
      val s = System.nanoTime()
      val localModel = broadcastModel.cloneModel()
      val localPredictor = broadcastPredictor.value.clonePredictor()
      val localEvaluator = broadcastEvaluator.value
      logger.info("total clone time is " + (System.nanoTime() - s) / 1e9 + "s")

      dataIter.map(batch => {
        val start = System.nanoTime()
        val result = localPredictor.imDetect(localModel, batch.data)
        logger.info(s"detect ${ batch.path } with time ${ (System.nanoTime() - start) / 1e9 }s")
        if (localEvaluator.isDefined) {
          localEvaluator.get.evaluateSingle(result, batch.getGtBoxes,
            batch.getGtClasses, batch.path)
        }
        (result, batch.getGtBoxes, batch.getGtClasses, batch.path)
      })
    }).collect()
    val totalTime = (System.nanoTime() - start) / 1e9
    logger.info(s"[Prediction] ${ output.length } in $totalTime seconds. Throughput is ${
      output.length / totalTime
    } record / sec")
    output
  }


}
