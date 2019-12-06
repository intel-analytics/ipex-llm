package com.intel.analytics.zoo.apps.model.inference.flink.Resnet50ImageClassification

import com.intel.analytics.zoo.pipeline.inference.InferenceModel

class Resnet50InferenceModel(var concurrentNum: Int = 1, modelType: String, checkpointBytes: Array[Byte], inputShape: Array[Int], ifReverseInputChannels: Boolean, meanValues: Array[Float], scale: Float) extends InferenceModel(concurrentNum) with Serializable {
  doLoadTF(null, modelType, checkpointBytes, inputShape, ifReverseInputChannels, meanValues, scale)
}