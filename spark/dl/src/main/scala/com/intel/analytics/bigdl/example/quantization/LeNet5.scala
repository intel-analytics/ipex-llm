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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import scopt.OptionParser

case class Params(
  modelSnapshot: Option[String] = None
)
object LeNet5 {
  val parser = new OptionParser[Params]("BigDL Vgg on Cifar10 Example") {
    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelSnapshot = Some(x)))
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, Params()).foreach {
      param => {
        val model = Module.load[Float](param.modelSnapshot.get)
        val quantizedModel = Module.quantize(model)

        val input = Tensor[Float](1, 1, 28, 28).fill(1.0f)

        for (i <- 0 until input.nElement()) {
          input.storage().array()(i) = i % 32
        }

        model.updateOutput(input)
        quantizedModel.updateOutput(input)

        println(model.output)
        println(quantizedModel.output)
        println(model.output.toTensor.max())
        println(quantizedModel.output.toTensor.max())

        println(model)
        println("=" * 80)
        println(quantizedModel)
      }
    }
  }
}
