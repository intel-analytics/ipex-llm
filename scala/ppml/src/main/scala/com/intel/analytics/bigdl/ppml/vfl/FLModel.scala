package com.intel.analytics.bigdl.ppml.vfl

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dllib.feature.dataset.MiniBatch
import com.intel.analytics.bigdl.ppml.FLClient
import com.intel.analytics.bigdl.ppml.vfl.nn.VflNNEstimator

class FLModel() {
  val estimator: VflNNEstimator = null
  def fit(trainData: DataSet[MiniBatch[Float]],
          valData: DataSet[MiniBatch[Float]],
          epoch : Int = 1) = {
    estimator.train(epoch, trainData.toLocal(), valData.toLocal())
  }
  def evaluate() = {
    estimator.getEvaluateResults().foreach{r =>
      println(r._1 + ":" + r._2.mkString(","))
    }
  }
  def predict(data: DataSet[MiniBatch[Float]]) = {

  }
}

