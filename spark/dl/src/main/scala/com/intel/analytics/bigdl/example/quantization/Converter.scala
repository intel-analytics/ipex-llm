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

package com.intel.analytics.bigdl.example.quantization

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.caffe.CaffeLoader
import scopt.OptionParser

object Converter {

  case class ConverterParam(
    modelType: String = "",
    from: String = "",
    prototxt: String = "",
    to: String = ""
  )

  val converterParser = new OptionParser[ConverterParam]("Convert caffe model to bigdl model") {
    opt[String]("modelType")
            .text("What's the type of origin model?")
            .action((x, c) => c.copy(modelType = x))
            .required()
    opt[String]("from")
            .text("Where's the origin model file?")
            .action((x, c) => c.copy(from = x))
            .required()
    opt[String]("prototxt")
            .text("Where's the caffe deploy prototxt?")
            .action((x, c) => c.copy(prototxt = x))
    opt[String]("to")
            .text("Where's the bigdl model?")
            .action((x, c) => c.copy(to = x))
            .required()
  }

  def main(args: Array[String]): Unit = {
    converterParser.parse(args, ConverterParam()).foreach { param =>
      val model = param.modelType.toLowerCase match {
        case "torch" =>
          Module.loadTorch(param.from)
        case "caffe" =>
          CaffeLoader.loadCaffe(param.prototxt, param.from)._1
      }

      model.save(param.to)
    }
  }
}
