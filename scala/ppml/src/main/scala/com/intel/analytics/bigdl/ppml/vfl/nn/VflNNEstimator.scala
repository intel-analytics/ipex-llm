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

package com.intel.analytics.bigdl.ppml.vfl.nn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dllib.feature.dataset.{LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.dllib.keras.models.InternalOptimizerUtil
import com.intel.analytics.bigdl.dllib.keras.models.InternalOptimizerUtil.getParametersFromModel
import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.optim.OptimMethod
import com.intel.analytics.bigdl.ppml.FLClient
import com.intel.analytics.bigdl.ppml.generated.FLProto.{EvaluateResponse, TableMetaData}
import com.intel.analytics.bigdl.ppml.vfl.VflContext
import com.intel.analytics.bigdl.ppml.vfl.utils.ProtoUtils._
import org.apache.log4j.Logger

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class VflNNEstimator(algorithm: String,
                     model: Module[Float],
                     optimMethod: OptimMethod[Float]){
  val logger = Logger.getLogger(getClass)
  val flClient = VflContext.getClient()
  val (weight, grad) = getParametersFromModel(model)

  def train(endEpoch: Int,
            trainDataSet: LocalDataSet[MiniBatch[Float]]): Module[Float] = {
    train(endEpoch, trainDataSet, null)
  }

  protected val evaluateResults = mutable.Map[String, ArrayBuffer[Float]]()

  def getEvaluateResults(): Map[String, Array[Float]] = {
    evaluateResults.map(v => (v._1, v._2.toArray)).toMap
  }

  def train(endEpoch: Int,
            trainDataSet: LocalDataSet[MiniBatch[Float]],
            valDataSet: LocalDataSet[MiniBatch[Float]]): Module[Float] = {
    val clientUUID = flClient.getClientUUID()
    val size = trainDataSet.size()
    var iteration = 0
    (0 until endEpoch).foreach {epoch =>
      val dataSet = trainDataSet.data(true)
      var count = 0
      while (count < size) {
        val miniBatch = dataSet.next()
        miniBatch.size()
        InternalOptimizerUtil.getStateFromOptiMethod(optimMethod)
          .update("epoch", epoch + 1)
        InternalOptimizerUtil.getStateFromOptiMethod(optimMethod)
          .update("neval", iteration + 1)
        val input = miniBatch.getInput()
        val target = miniBatch.getTarget()
        model.training()
        val output = model.forward(input)

        // Upload to PS
        uploadOutput(model, iteration, target)
        model.zeroGradParameters()
        // Download average model
        val gradInput = downloadTrain(flClient, "gradInput", iteration, algorithm)
        // model replace
        val errors = getTensor("gradInput", gradInput)
        val loss = getTensor("loss", gradInput).value()
        model.backward(input, errors)

        optimMethod.optimize(_ => (loss, grad), weight)

        iteration += 1
        count += miniBatch.size()
      }
      model.evaluate()
      val valIterator = valDataSet.data(false)
      var evaluateResponse: EvaluateResponse = null;
      while(valIterator.hasNext) {
        val miniBatch = valIterator.next()
        val input = miniBatch.getInput()
        val target = miniBatch.getTarget()
        val output = model.forward(input)
        evaluateResponse = evaluateOutput(model, epoch + 1, target, !valIterator.hasNext)
      }
      logger.info(evaluateResponse.getResponse)
      val dataMap = evaluateResponse.getData.getTableMap.asScala
      dataMap.foreach{v =>
        if (evaluateResults.contains(v._1)) {
          evaluateResults(v._1).append(v._2.getTensor(0))
        } else {
          evaluateResults(v._1) = ArrayBuffer(v._2.getTensor(0))
        }
      }
    }

    model
  }

  def close(): Unit = {
    flClient.shutdown()
  }
  def uploadOutput(model: Module[Float], flVersion: Int, target: Activity = null): Unit = {
    val metadata = TableMetaData.newBuilder
      .setName(s"${model.getName()}_output").setVersion(flVersion).build

    // TODO: support table output and table target
    val tableProto = outputTargetToTableProto(model.output, target, metadata)
    flClient.nnStub.uploadTrain(tableProto, algorithm)
  }

  def evaluateOutput(model: Module[Float],
                     flVersion: Int,
                     target: Activity,
                     lastBatch: Boolean): EvaluateResponse = {
    val metadata = TableMetaData.newBuilder
      .setName(s"${model.getName()}_output").setVersion(flVersion).build

    // TODO: support table output and table target
    val tableProto = outputTargetToTableProto(model.output, target, metadata)
    flClient.nnStub.evaluate(tableProto, lastBatch)
  }

}

