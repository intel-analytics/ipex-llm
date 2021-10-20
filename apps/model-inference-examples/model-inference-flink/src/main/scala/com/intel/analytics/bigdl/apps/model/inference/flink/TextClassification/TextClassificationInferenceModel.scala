package com.intel.analytics.bigdl.apps.model.inference.flink.TextClassification

import com.intel.analytics.bigdl.serving.pipeline.inference.InferenceModel

class TextClassificationInferenceModel(val supportedConcurrentNum: Int, val stopWordsCount: Int, val sequenceLength: Int, val embeddingFilePath: String)
  extends InferenceModel(supportedConcurrentNum) with Serializable {

  val textProcessor = new TextProcessor(stopWordsCount, sequenceLength, embeddingFilePath)

  def preprocess(text: String) = textProcessor.preprocess(text)
}
