package com.intel.analytics.bigdl.ppml.base

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dllib.feature.dataset.{LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.optim.Metrics
import com.intel.analytics.bigdl.dllib.tensor.Tensor

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

trait Estimator {
  protected val evaluateResults: mutable.Map[String, ArrayBuffer[Float]]
  def getEvaluateResults(): Map[String, Array[Float]] = {
    evaluateResults.map(v => (v._1, v._2.toArray)).toMap
  }
  def train(endEpoch: Int,
            trainDataSet: LocalDataSet[MiniBatch[Float]],
            valDataSet: LocalDataSet[MiniBatch[Float]]): Module[Float]

  def evaluate(dataSet: LocalDataSet[MiniBatch[Float]])

  def predict(dataSet: LocalDataSet[MiniBatch[Float]]): Array[Activity]
}
