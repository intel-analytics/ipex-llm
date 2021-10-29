package com.intel.analytics.bigdl.ppml.vfl

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dllib.feature.dataset.MiniBatch
import com.intel.analytics.bigdl.dllib.nn.{Linear, Sequential}
import com.intel.analytics.bigdl.dllib.optim.Adam
import com.intel.analytics.bigdl.ppml.vfl.nn.VflNNEstimator

class LogisticRegression(featureNum: Int,
                         learningRate: Float = 0.005f) extends VflModel() {
  val model = Sequential[Float]().add(Linear(featureNum, 1))
  override val estimator = new VflNNEstimator(
    "logistic_regression", model, new Adam(learningRate))

}
