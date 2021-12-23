package com.intel.analytics.bigdl.ppml.fgboost.common

trait TreeObjective {
  def getGradient(predict: Array[Float],
                  label: Array[Float]): Array[Array[Float]]

  def getLoss(predict: Array[Float],
              label: Array[Float]): Float

}
