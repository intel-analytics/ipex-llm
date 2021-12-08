package com.intel.analytics.bigdl.ppml.algorithms.hfl

import com.intel.analytics.bigdl.dllib.nn.{Linear, Sequential}
import com.intel.analytics.bigdl.dllib.optim.Adam
import com.intel.analytics.bigdl.ppml.FLModel
import com.intel.analytics.bigdl.ppml.hfl.nn.HflNNEstimator
import com.intel.analytics.bigdl.ppml.utils.FLClientClosable

class LogisticRegression(featureNum: Int,
                         learningRate: Float = 0.005f) extends FLModel() with FLClientClosable {
  val model = Sequential[Float]().add(Linear(featureNum, 1))
  override val estimator = new HflNNEstimator(
    "logistic_regression", model, new Adam(learningRate))
