package com.intel.analytics.bigdl.ppml.algorithms.vfl

import com.intel.analytics.bigdl.dllib.nn.Sequential
import com.intel.analytics.bigdl.ppml.FLModel
import com.intel.analytics.bigdl.ppml.base.Estimator
import com.intel.analytics.bigdl.ppml.vfl.fgboost.VflGBoostEstimator

class FGBoostRegression(learningRate: Float = 0.005f,
                        maxDepth: Int = 6,
                        minChildSize: Int = 1) extends FLModel {
  override val model: Sequential[Float] = null
  override val estimator: Estimator = new VflGBoostEstimator(learningRate, maxDepth, minChildSize)
}
