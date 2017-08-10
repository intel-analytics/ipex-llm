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
import scopt.OptionParser

object ConvertModel {

  case class ConverterParam(
    from: String = "",
    to: String = "",
    input: String = "",
    output: String = "",
    prototxt: String = "",
    quantize: Boolean = false
  )

  val converterParser = new OptionParser[ConverterParam]("Convert caffe model to bigdl model") {
    opt[String]("from")
            .text("What's the type origin model (caffe, torch, bigdl)?")
            .action((x, c) => c.copy(from = x))
            .required()
    opt[String]("to")
            .text("What's the type of model you want (bigdl, torch)?")
            .action((x, c) => c.copy(to = x))
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
            .text("Do you want to quantize the model?")
            .action((x, c) => c.copy(quantize = x))
  }

  def main(args: Array[String]): Unit = {
    converterParser.parse(args, ConverterParam()).foreach { param =>
      val input = param.input
      val output = param.output

      val loadedModel = param.from.toLowerCase match {
        case "bigdl" =>
          Module.load[Float](input)
        case "torch" =>
          Module.loadTorch[Float](input)
        case "caffe" =>
          CaffeLoader.loadCaffe[Float](param.prototxt, input)._1
        case _ =>
          throw new UnsupportedOperationException(s"Unsupported model type.")
      }

      val model = if (param.quantize) {
        Module.quantize[Float](loadedModel)
      } else {
        loadedModel
      }

      param.to.toLowerCase match {
        case "bigdl" =>
          model.save(output)
        case "torch" =>
          model.saveTorch(output)
        case "caffe" =>
          model.saveCaffe(param.prototxt, output)
        case _ =>
          throw new UnsupportedOperationException(s"Unsupported model type.")
      }
    }
  }
}
