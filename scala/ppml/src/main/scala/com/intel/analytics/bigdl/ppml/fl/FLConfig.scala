package com.intel.analytics.bigdl.ppml.fl

class FLConfig {
  var modelPath: String = null
  def setModelPath(modelPath: String): Unit = {
    this.modelPath = modelPath
  }
}
