package com.intel.analytics.bigdl.ppml.hfl.nn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dllib.estimator.LocalEstimator
import com.intel.analytics.bigdl.dllib.feature.dataset.{LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.dllib.optim.OptimMethod
import com.intel.analytics.bigdl.ppml.FLContext
import com.intel.analytics.bigdl.ppml.base.Estimator
import com.intel.analytics.bigdl.ppml.utils.ProtoUtils._
import org.apache.log4j.Logger


class HflNNEstimator(algorithm: String,
                     model: Module[Float],
                     optimMethod: OptimMethod[Float],
                     threadNum: Int = 1) extends Estimator{
  val logger = Logger.getLogger(getClass)
  val flClient = FLContext.getClient()
  val localEstimator = LocalEstimator(model = model, criterion = null, optmizeMethod = optimMethod,
    null, threadNum)
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
      uploadModel(flClient, model, iteration, algorithm)
      // Download average model
      val newModel = downloadTrain(flClient, "test", iteration, algorithm)
      // model replace
      updateModel(model, newModel)
      iteration += 1

    }

    model
  }
}
