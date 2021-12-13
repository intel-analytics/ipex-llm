package com.intel.analytics.bigdl.ppml.hfl.nn

import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.bigdl.dllib.estimator.LocalEstimator
import com.intel.analytics.bigdl.dllib.feature.dataset.{LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.dllib.optim.{LocalPredictor, Metrics, OptimMethod, ValidationMethod}
import com.intel.analytics.bigdl.ppml.FLContext
import com.intel.analytics.bigdl.ppml.base.Estimator
import com.intel.analytics.bigdl.ppml.utils.ProtoUtils._
import org.apache.log4j.Logger

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer


/**
 * @param algorithm algorithm of NN, supported:
 *                  hfl_logistic_regression, hfl_linear_regression
 * @param model model
 * @param optimMethod optimizer
 * @param criterion loss function, HFL takes loss at estimator, VFL takes loss at aggregator
 * @param threadNum
 */
class HflNNEstimator(algorithm: String,
                     model: Module[Float],
                     optimMethod: OptimMethod[Float],
                     criterion: Criterion[Float],
                     metrics: Array[ValidationMethod[Float]] = null,
                     threadNum: Int = 1) extends Estimator{
  val logger = Logger.getLogger(getClass)
  val flClient = FLContext.getClient()
  val localEstimator = LocalEstimator(
    model = model, criterion = criterion, optmizeMethod = optimMethod, null, threadNum)
  val localPredictor = LocalPredictor[Float](model)
  protected val evaluateResults = mutable.Map[String, ArrayBuffer[Float]]()


  def train(endEpoch: Int,
            trainDataSet: LocalDataSet[MiniBatch[Float]],
            valDataSet: LocalDataSet[MiniBatch[Float]]): Module[Float] = {
    val clientUUID = flClient.getClientUUID()
    val size = trainDataSet.size()
    var iteration = 0
    (0 until endEpoch).foreach { epoch =>
      val trainSet = trainDataSet.data(true)
      val valSet = valDataSet.data(false)
      localEstimator.fit(trainSet.toSeq, size.toInt, valSet.toSeq)
      logger.debug(s"Local train step ends, syncing version: $iteration with server.")
      val weights = getModelWeightTable(model, iteration)
      val serverWeights = flClient.nnStub.train(weights, algorithm).getData

      // model replace
      updateModel(model, serverWeights)
      logger.debug(s"Local tensor updated from server version.")
      iteration += 1

    }

    model
  }
  def evaluate(dataSet: LocalDataSet[MiniBatch[Float]]) = {
    model.evaluate(dataSet, metrics)
  }
  def predict(dataSet: LocalDataSet[MiniBatch[Float]]) = {
    localPredictor.predict(dataSet)
  }
}
