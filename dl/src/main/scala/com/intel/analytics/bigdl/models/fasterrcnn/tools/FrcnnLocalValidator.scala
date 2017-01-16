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

import com.intel.analytics.bigdl.dataset.LocalDataSet
import com.intel.analytics.bigdl.models.fasterrcnn.dataset.Target
import com.intel.analytics.bigdl.models.fasterrcnn.dataset.transformers.MiniBatch
import com.intel.analytics.bigdl.models.fasterrcnn.model.FasterRcnn
import com.intel.analytics.bigdl.models.fasterrcnn.tools.FrcnnLocalValidator._
import com.intel.analytics.bigdl.models.fasterrcnn.utils.PascalVocEvaluator
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, MklBlas}
import org.apache.log4j.Logger

object FrcnnLocalValidator {
  val logger = Logger.getLogger(getClass)
}

class FrcnnLocalValidator(net: FasterRcnn, classNum: Int,
  dataSet: LocalDataSet[MiniBatch],
  maxPerImage: Int = 100, thresh: Double = 0.05) {
  val model = net.getTestModel
  val predParam = PredictorParam(net.param.NMS, classNum, net.param.BBOX_VOTE,
    maxPerImage, thresh)
  val predictor = new Predictor(predParam)

  def assertEngineInited(): Unit = {
    require(Engine.isInitialized, s"you may forget to initialize Engine.")
  }

  private val coreNumber = Engine.coreNumber()

  private val subModelNumber = Engine.getEngineType() match {
    case MklBlas => coreNumber
    case _ => throw new IllegalArgumentException
  }
  private val workingModels = (1 to subModelNumber).map(_ => model.cloneModule().evaluate()).toArray
  private val predictors = (1 to subModelNumber).map(_ => predictor.clonePredictor())

  def test(evaluator: Option[PascalVocEvaluator] = None):
  Array[(Array[Target], Tensor[Float], Tensor[Float], String)] = {
    this.assertEngineInited()
    val dataIter = dataSet.data(train = false)
    var count = 0
    val images = (1 to subModelNumber).map(i => new MiniBatch)

    val splitNum = Math.ceil(dataSet.size() / subModelNumber.toFloat).toInt
    var output = Array[(Array[Target], Tensor[Float], Tensor[Float], String)]()
    var s = 0
    while (s < splitNum) {
      val start = System.nanoTime()
      val num = if (s == splitNum - 1) dataSet.size().toInt - subModelNumber * s else subModelNumber
      var i = 0
      while (i < num) {
        images(i).copy(dataIter.next())
        i += 1
      }
      val results = Engine.fixed.invokeAndWait(
        (0 until num).map(b => () => {
          val batch = images(b)
          val start = System.nanoTime()
          val result = predictors(b).imDetect(workingModels(b), batch.data)
          logger.info(s"detect ${ batch.path } with time ${ (System.nanoTime() - start) / 1e9 }s")
          if (evaluator.isDefined) {
            evaluator.get.evaluateSingle(result, batch.getGtBoxes, batch.getGtClasses)
          }
          (result, batch.getGtBoxes, batch.getGtClasses, batch.path)
        }))
      output = output ++ results
      count += num
      logger.info(s"[Prediction] $count/${ dataSet.size() } Throughput is ${
        num / ((System.nanoTime() - start) / 1e9)
      } record / sec")
      s += 1
    }
    output
  }
}
