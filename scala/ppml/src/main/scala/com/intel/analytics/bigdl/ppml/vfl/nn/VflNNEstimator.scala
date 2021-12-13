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
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.ppml.base.Estimator
import com.intel.analytics.bigdl.ppml.generated.FlBaseProto._
import com.intel.analytics.bigdl.ppml.generated.NNServiceProto.EvaluateResponse
import com.intel.analytics.bigdl.ppml.{FLClient, FLContext}
import com.intel.analytics.bigdl.ppml.utils.ProtoUtils._
import org.apache.log4j.Logger

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class VflNNEstimator(algorithm: String,
                     model: Module[Float],
                     optimMethod: OptimMethod[Float],
                     threadNum: Int = 1) extends Estimator {
  val logger = Logger.getLogger(getClass)
  val flClient = FLContext.getClient()
  val (weight, grad) = getParametersFromModel(model)

  def train(endEpoch: Int,
            trainDataSet: LocalDataSet[MiniBatch[Float]]): Module[Float] = {
    train(endEpoch, trainDataSet, null)
  }

  protected val evaluateResults = mutable.Map[String, ArrayBuffer[Float]]()

  def train(endEpoch: Int,
            trainDataSet: LocalDataSet[MiniBatch[Float]],
            valDataSet: LocalDataSet[MiniBatch[Float]]): Module[Float] = {
    val clientUUID = flClient.getClientUUID()
    val size = trainDataSet.size()
    var iteration = 0
    (0 until endEpoch).foreach { epoch =>
      val dataSet = trainDataSet.data(true)
      var count = 0
      while (count < size) {
        logger.debug(s"training next batch, progress: $count/$size, epoch: $epoch/$endEpoch")
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
        val metadata = TableMetaData.newBuilder
              .setName(s"${model.getName()}_output").setVersion(iteration).build
        val tableProto = outputTargetToTableProto(model.output, target, metadata)
        model.zeroGradParameters()
        val gradInput = flClient.nnStub.train(tableProto, algorithm).getData

        // model replace
        val errors = getTensor("gradInput", gradInput)
        val loss = getTensor("loss", gradInput).value()
        model.backward(input, errors)
        logger.debug(s"Model doing backward, version: $iteration")
        optimMethod.optimize(_ => (loss, grad), weight)

        iteration += 1
        count += miniBatch.size()
      }
      if (valDataSet != null) {
        model.evaluate()
        evaluate(valDataSet)
//        val valIterator = valDataSet.data(false)
//        var evaluateResponse: EvaluateResponse = null;
//        while(valIterator.hasNext) {
//          val miniBatch = valIterator.next()
//          val input = miniBatch.getInput()
//          val target = miniBatch.getTarget()
//          val output = model.forward(input)
//          val metadata = TableMetaData.newBuilder
//            .setName(s"${model.getName()}_output").setVersion(iteration).build
//
//          // TODO: support table output and table target
//          val tableProto = outputTargetToTableProto(output, target, metadata)
//          evaluateResponse = flClient.nnStub.evaluate(tableProto, algorithm, false)
//        }
//        logger.info(evaluateResponse.getResponse)
//        val dataMap = evaluateResponse.getData.getTableMap.asScala
//        dataMap.foreach{v =>
//          if (evaluateResults.contains(v._1)) {
//            evaluateResults(v._1).append(v._2.getTensor(0))
//          } else {
//            evaluateResults(v._1) = ArrayBuffer(v._2.getTensor(0))
//          }
//        }
      }

    }

    model
  }

  /**
   * Evaluate VFL model
   * For each batch, client estimator upload output tensor to server aggregator,
   * the server keep updating the validation result until last batch of data,
   * then return the evaluation result
   * @param dataSet the data to evaluate
   */
  override def evaluate(dataSet: LocalDataSet[MiniBatch[Float]]): Unit = {
    val dataSize = dataSet.size()
    val data = dataSet.data(false)
    var count = 0
    var iteration = 0
    var evaluateResult = ""
    while (count < dataSize) {
      logger.debug(s"evaluating next batch, progress: $count/$dataSize")
      model.evaluate()
      val miniBatch = data.next()
      val input = miniBatch.getInput()
      val target = miniBatch.getTarget()
      val output = model.forward(input)

      val metadata = TableMetaData.newBuilder
        .setName(s"${model.getName()}_output").setVersion(iteration).build
      val tableProto = outputTargetToTableProto(model.output, target, metadata)
      val hasReturn = if (!data.hasNext) true else false
      evaluateResult = flClient.nnStub.evaluate(tableProto, algorithm, hasReturn).getMessage
      iteration += 1
      count += miniBatch.size()
    }
    println(s"Evaluation result: $evaluateResult")
  }

  override def predict(dataSet: LocalDataSet[MiniBatch[Float]]): Array[Activity] = {
    val dataSize = dataSet.size()
    val data = dataSet.data(false)
    var count = 0
    var iteration = 0
    var resultSeq = Seq[Tensor[Float]]()
    while (count < dataSize) {
      model.evaluate()
      val miniBatch = data.next()
      val input = miniBatch.getInput()
      val target = miniBatch.getTarget()
      val output = model.forward(input)

      val metadata = TableMetaData.newBuilder
        .setName(s"${model.getName()}_output").setVersion(iteration).build
      val tableProto = outputTargetToTableProto(model.output, target, metadata)
      val hasReturn = if (!data.hasNext) true else false
      val result = flClient.nnStub.predict(tableProto, algorithm).getData
      resultSeq = resultSeq :+ getTensor("predictOutput", result)
      iteration += 1
      count += miniBatch.size()
    }
    resultSeq.toArray
  }

  def close(): Unit = {
    flClient.shutdown()
  }

}
