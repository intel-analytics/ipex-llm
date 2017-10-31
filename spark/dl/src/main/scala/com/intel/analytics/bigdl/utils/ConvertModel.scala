/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.utils

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.utils.caffe.CaffeLoader
import com.intel.analytics.bigdl.numeric.NumericFloat
import scopt.OptionParser

object ConvertModel {

  case class ConverterParam(
    from: String = "",
    to: String = "",
    input: String = "",
    output: String = "",
    prototxt: String = "",
    tf_inputs: String = "",
    tf_outputs: String = "",
    quantize: Boolean = false
  )

  val fromSupports = Set("bigdl", "caffe", "torch", "tensorflow")
  val toSupports = Set("bigdl", "caffe", "torch")

  val converterParser = new OptionParser[ConverterParam](
    "Convert models between different dl frameworks") {
    opt[String]("from")
      .text(s"What's the type origin model ${fromSupports.mkString(",")}?")
      .action((x, c) => c.copy(from = x))
      .validate(x =>
        if (fromSupports.contains(x.toLowerCase)) {
          success
        } else {
          failure(s"Only support ${fromSupports.mkString(",")}")
        })
      .required()
    opt[String]("to")
      .text(s"What's the type of model you want ${toSupports.mkString(",")}?")
      .action((x, c) => c.copy(to = x))
      .validate(x =>
        if (toSupports.contains(x.toLowerCase)) {
          success
        } else {
          failure(s"Only support ${toSupports.mkString(",")}")
        })
      .required()
    opt[String]("input")
      .text("Where's the origin model file?")
      .action((x, c) => c.copy(input = x))
      .required()
    opt[String]("output")
      .text("Where's the bigdl model file to save?")
      .action((x, c) => c.copy(output = x))
      .required()
    opt[String]("prototxt")
      .text("Where's the caffe deploy prototxt?")
      .action((x, c) => c.copy(prototxt = x))
    opt[Boolean]("quantize")
      .text("Do you want to quantize the model? Only works when \"--to\" is bigdl;" +
        "you can only perform inference using the new quantized model.")
      .action((x, c) => c.copy(quantize = x))
    opt[String]("tf_inputs")
      .text("Inputs for Tensorflow")
      .action((x, c) => c.copy(tf_inputs = x))
    opt[String]("tf_outputs")
      .text("Outputs for Tensorflow")
      .action((x, c) => c.copy(tf_outputs = x))

    checkConfig(c =>
      if (c.from.toLowerCase == "caffe" && c.prototxt.isEmpty) {
        failure(s"If model is converted from caffe, the prototxt should be given with --prototxt.")
      } else if (c.from.toLowerCase == "tensorflow" &&
        (c.tf_inputs.isEmpty || c.tf_outputs.isEmpty)) {
        failure(s"If model is converted from tensorflow, inputs and outputs should be given")
      } else if (c.quantize == true && c.to.toLowerCase != "bigdl") {
        failure(s"Only support quantizing models to BigDL model now.")
      } else {
        success
      }
    )
  }

  def main(args: Array[String]): Unit = {
    converterParser.parse(args, ConverterParam()).foreach { param =>
      val input = param.input
      val output = param.output
      val ifs = ","

      var loadedModel = param.from.toLowerCase match {
        case "bigdl" =>
          Module.loadModule(input)
        case "torch" =>
          Module.loadTorch(input)
        case "caffe" =>
          CaffeLoader.loadCaffe(param.prototxt, input)._1
        case "tensorflow" =>
          val inputs = param.tf_inputs.split(ifs)
          val outputs = param.tf_outputs.split(ifs)
          Module.loadTF(input, inputs, outputs)
      }

      val model = if (param.quantize) {
        loadedModel.quantize()
      } else {
        loadedModel
      }

      param.to.toLowerCase match {
        case "bigdl" =>
          model.saveModule(output, overWrite = true)
        case "torch" =>
          model.saveTorch(output)
        case "caffe" =>
          model.saveCaffe(param.prototxt, output)
      }
    }
  }
}
