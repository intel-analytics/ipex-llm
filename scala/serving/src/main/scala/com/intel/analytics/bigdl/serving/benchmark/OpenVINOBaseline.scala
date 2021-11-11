package com.intel.analytics.bigdl.serving.benchmark

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.orca.inference.InferenceModel
import com.intel.analytics.bigdl.serving.ClusterServingHelper
import scopt.OptionParser

object OpenVINOBaseline {
  case class Config(modelPath: String = null)
  val parser = new OptionParser[Config]("DIEN benchmark test Usage") {
    opt[String]('m', "modelPath")
      .text("Model Path for Test")
      .action((x, params) => params.copy(modelPath = x))
      .required()
  }
  def main(args: Array[String]): Unit = {

    val helper = new ClusterServingHelper()
    val arg = parser.parse(args, Config()).head
    helper.modelPath = arg.modelPath
    val model = helper.loadInferenceModel()
    (0 until 10).foreach(_ => {
      val t = Tensor[Float](4, 3, 224, 224).rand()
      model.doPredict(t)
    })

  }
}
