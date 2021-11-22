package com.intel.analytics.bigdl.ppml.algorithms.vfl

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dllib.feature.dataset.MiniBatch
import com.intel.analytics.bigdl.dllib.nn.{Linear, Sequential}
import com.intel.analytics.bigdl.dllib.optim.Adam
import com.intel.analytics.bigdl.ppml.vfl.{VflNNEstimator, VflModel}

class LogisticRegression(featureNum: Int,
                         learningRate: Float = 0.005f) extends VflModel() {
  val model = Sequential[Float]().add(Linear(featureNum, 1))
  val estimator = new VflNNEstimator(flClient, model, new Adam(learningRate))
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
