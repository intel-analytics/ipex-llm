package com.intel.analytics.bigdl.ppml.utils

import com.intel.analytics.bigdl.dllib.feature.dataset.{LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.dllib.tensor.Tensor

import scala.collection.mutable.ArrayBuffer

object DataSetUtils {
  def localDataSetToArray(dataSet: LocalDataSet[MiniBatch[Float]]) = {
    val featureBuffer = new ArrayBuffer[Tensor[Float]]()
    val labelBuffer = new ArrayBuffer[Float]()
    var count = 0
    val data = dataSet.data(true)
    while (count < data.size) {
      val batch = data.next()
      featureBuffer.append(batch.getInput().toTensor[Float])
      labelBuffer.append(batch.getTarget().toTensor[Float].value())
    }
    (featureBuffer.toArray, labelBuffer.toArray)
  }
}
