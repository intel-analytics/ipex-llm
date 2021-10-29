package com.intel.analytics.bigdl.ppml.vfl

import com.intel.analytics.bigdl.dllib.nn.{Linear, Sequential}
import com.intel.analytics.bigdl.dllib.optim.Adam
import com.intel.analytics.bigdl.ppml.vfl.nn.VflNNEstimator

class LinearRegression(featureNum: Int,
                       learningRate: Float = 0.005f) extends VflModel{
  val model = Sequential[Float]().add(Linear(featureNum, 1))
  override val estimator = new VflNNEstimator(
    "linear_regression", model, new Adam(learningRate))
}
