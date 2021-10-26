package com.intel.analytics.bigdl.ppml.vfl.example



object ExampleUtils {
  def minMaxNormalize(data: Array[Array[Float]], col: Int): Array[Array[Float]] = {
    val min = data.map(_ (col)).min
    val max = data.map(_ (col)).max
    data.foreach { d =>
      d(col) = (d(col) - min) / (max - min)
    }
    data
  }
}
